# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer-based machine translation model."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
# See issue #620.
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

from functools import partial
from typing import Any, Callable, Tuple, Optional

import jax
from flax import linen as nn
from flax import struct
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np

import flax.linen.linear as fll 
import flax.linen.module as flm 
import flax.linen.initializers as fli

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None
  sinusoidal: bool = False
  relative_radius: Optional[int] = None
  relative_bias: Optional[bool] = False
  enc2dec: Optional[bool] = False
  copy_decoder: bool = False


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[:, :-1]


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init
  

def create_relative_ids(in_length, relative_radius, tar_length1=None, 
                        tar_length2=None, dec2enc_ids=None, cache_idx=jnp.zeros(1)):
  """Creates 2D Relative IDs for Relative Position Encodings. Relative ID matrices
     are Toeplitz matrices with shape with shape (d1, d2).

  Args:
      in_length: int: maximum possible length for the input.
      relative_radius: int: relative radius for relative attention.
      tar_length1: int: maximum possible length for the output currently being 
        decoded (when decode=True, tar_length1 is different then tar_length2).
      tar_length1: int: maximum possible length for the output.
      dec2enc_ids: bool: whether to return a Toeplitz matrix (True) or a constant
        matrix equal to relative_radius (False) for the decoder-encoder IDs.
      cache_idx: index of the current output position (used when decode=True).

  Returns:
      output: encoder relative IDs with shape (in_length, in_length) when 
        tar_length1 is None; encoder relative IDs with shape (in_length, in_length),
        decoder relative IDs with shape (tar_length1, tar_length2) and 
        decoder-to-encoder relative IDs with shape (tar_length1, in_length) otherwise.
  """

  indices = np.arange(in_length)
  diff = np.subtract(np.expand_dims(indices, 1), np.expand_dims(indices, 0))
  diff = jnp.array(diff) 
  enc_relative_ids = relative_radius 
  enc_relative_ids += jnp.minimum(jnp.maximum(diff, -relative_radius), relative_radius)
  enc_relative_ids = jnp.array(enc_relative_ids, dtype=int)
  if tar_length1 is not None:
    if tar_length2 is None:
      tar_length2 = tar_length1
    indices1 = np.arange(tar_length1)
    indices2 = np.arange(tar_length2)
    diff = np.subtract(np.expand_dims(indices1, 1), np.expand_dims(indices2, 0))
    diff = jnp.array(diff) + cache_idx
    dec_relative_ids = relative_radius + jnp.minimum(jnp.maximum(diff, 
                                            -relative_radius), relative_radius)
    dec_relative_ids = jnp.array(dec_relative_ids, dtype=int)
    if dec2enc_ids:
      indices1 = np.arange(tar_length1)
      indices2 = np.arange(in_length)
      diff = np.subtract(np.expand_dims(indices1, 1), np.expand_dims(indices2, 0))
      diff = jnp.array(diff) + cache_idx
      dec2enc_relative_ids = relative_radius 
      dec2enc_relative_ids += jnp.minimum(jnp.maximum(diff, -relative_radius), 
                                          relative_radius)
    else:
      dec2enc_relative_ids = jnp.ones([tar_length1, in_length], dtype=int) 
      dec2enc_relative_ids *= relative_radius
    dec2enc_relative_ids = jnp.array(dec2enc_relative_ids, dtype=int)
    return enc_relative_ids, dec_relative_ids, dec2enc_relative_ids
  else:
    return enc_relative_ids

# TODO: Create unit tests for create_relative_ids

class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """
  config: TransformerConfig
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    cfg = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, cfg.max_len, inputs.shape[-1])
    if cfg.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=cfg.max_len)(
          None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 cfg.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    cfg = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(cfg.mlp_dim,
                 dtype=cfg.dtype,
                 kernel_init=cfg.kernel_init,
                 bias_init=cfg.bias_init)(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    output = nn.Dense(actual_out_dim,
                      dtype=cfg.dtype,
                      kernel_init=cfg.kernel_init,
                      bias_init=cfg.bias_init)(x)
    output = nn.Dropout(rate=cfg.dropout_rate)(
        output, deterministic=cfg.deterministic)
    return output


def dot_product_relative_attention_weights(query: Array,
                                  key: Array,
                                  bias: Optional[Array] = None,
                                  relative_ids: Optional[Array] = None,
                                  relative_embeddings: Optional[Callable] = None,
                                  relative_biases: Optional[Callable] = None,
                                  broadcast_dropout: bool = True,
                                  dropout_rng: Optional[PRNGKey] = None,
                                  dropout_rate: float = 0.,
                                  deterministic: bool = False,
                                  dtype: Dtype = jnp.float32,
                                  precision: Optional[lax.Precision] = None):
  """Computes dot-product attention weights given query and key.
  
  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, num_heads, qk_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks, padding masks,
      proximity bias, etc.
    relative_ids: relative ids used to calculate relative position encodings with 
      shape of `[q_length, kv_length]`.
    relative_embeddings: Callable: function used to calculate relative position
      encodings from relative_ids.
    relative_biases: Callable: function used to calculate relative bias from 
      relative_ids.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], (
      'q, k batch dims must match.')
  assert query.shape[-2] == key.shape[-2], (
      'q, k num_heads must match.')
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key,
                            precision=precision)

  if relative_ids is not None:
    if relative_embeddings is not None:
        r = relative_embeddings(relative_ids)
        matmul_qrel = jnp.einsum('...qhd,...qkd->...hqk', query, r, 
                                precision=precision)
        attn_weights += matmul_qrel

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  if relative_biases is not None:
    attn_weights += jnp.squeeze(relative_biases(relative_ids), axis = -1)
  
  attn_weights = attn_weights / jnp.sqrt(depth).astype(dtype)
  
  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = (keep.astype(attn_weights.dtype) /
                  jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  return attn_weights


def dot_product_relative_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          relative_ids: Optional[Array] = None,
                          relative_embeddings: Optional[Callable] = None,
                          relative_biases: Optional[Callable] = None,
                          broadcast_dropout: bool = True,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: Dtype = jnp.float32,
                          precision: Optional[lax.Precision] = None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of
      `[batch..., kv_length, num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks, padding masks,
      proximity bias, etc.
    relative_ids: relative ids used to calculate relative position encodings with 
      shape of `[q_length, kv_length]`.
    relative_embeddings: Callable: function used to calculate relative position
      encodings from relative_ids.
    relative_biases: Callable: function used to calculate relative bias from 
      relative_ids.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = dot_product_relative_attention_weights(
    query, key, bias, relative_ids, relative_embeddings,
    relative_biases, broadcast_dropout, dropout_rng, dropout_rate,
    deterministic, dtype, precision)

  # return weighted sum over values for each query position
  return jnp.einsum('...hqk,...khd->...qhd', attn_weights, value,
                    precision=precision)


class MultiHeadDotProductRelativeAttention(flm.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
      relative_radius: relative attention radius.
      relative_bias: bool: whether to add relative bias to attention matrix.
  """
  num_heads: int
  dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = fll.default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = fli.zeros
  use_bias: bool = True
  attention_fn: Callable[[Array, Array, Array], Array] = dot_product_relative_attention
  decode: bool = False
  relative_radius: Optional[int] = None
  relative_bias: bool = False

  @flm.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               relative_ids: Optional[Array] = None,
               mask: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      relative_ids: relative ids used to calculate relative position encodings with 
        shape of `[q_length, kv_length]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      deterministic = flm.merge_param('deterministic', self.deterministic, deterministic)
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = partial(fll.DenseGeneral,
                    axis=-1,
                    features=(self.num_heads, head_dim),
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    use_bias=self.use_bias,
                    precision=self.precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(dtype=self.dtype, name='query')(inputs_q),
                         dense(dtype=self.dtype, name='key')(inputs_kv),
                         dense(dtype=self.dtype, name='value')(inputs_kv))

    relative_embeddings = None
    relative_biases = None
    if self.relative_radius is not None:
      relative_vocab_size = 2 * self.relative_radius + 1
      head_dim = self.qkv_features // self.num_heads
      relative_embeddings = nn.Embed(relative_vocab_size, head_dim, 
                                          embedding_init=nn.initializers.normal(stddev=1.0))
      if self.relative_bias:
        relative_biases = nn.Embed(relative_vocab_size, 1,
                                        embedding_init=nn.initializers.normal(stddev=1.0))

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key',
                                 jnp.zeros, key.shape, key.dtype)
      cached_value = self.variable('cache', 'cached_value',
                                   jnp.zeros, value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = (
            cached_key.value.shape)
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = nn.combine_masks(
            mask,
            jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                             tuple(batch_dims) + (1, 1, max_length)))

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # apply attention
    x = self.attention_fn(
        query,
        key,
        value,
        bias=attention_bias,
        relative_ids=relative_ids,
        relative_embeddings=relative_embeddings,
        relative_biases=relative_biases,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args

    # back to the original inputs dimensions
    out = fll.DenseGeneral(features=features,
                       axis=(-2, -1),
                       kernel_init=self.kernel_init,
                       bias_init=self.bias_init,
                       use_bias=self.use_bias,
                       dtype=self.dtype,
                       precision=self.precision,
                       name='out')(x)
    return out

class SelfRelativeAttention(MultiHeadDotProductRelativeAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @flm.compact
  def __call__(self, inputs_q: Array, relative_ids: Optional[Array] = None,
               mask: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    return super().__call__(inputs_q, inputs_q, relative_ids, mask, 
                            deterministic=deterministic)

class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    relative_radius: relative attention radius.
  """
  config: TransformerConfig
  relative_radius: Optional[int] = None

  @nn.compact
  def __call__(self,
               inputs,
               relative_ids=None,
               encoder_mask=None):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      relative_ids: relative ids used to calculate relative position encodings.
      encoder_mask: encoder self-attention mask.

    Returns:
      output after transformer encoder block.
    """
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    x = SelfRelativeAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic,
        relative_radius=self.relative_radius,
        relative_bias=cfg.relative_bias)(x, relative_ids=relative_ids, 
                                             mask=encoder_mask)

    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    y = MlpBlock(config=cfg)(y)

    return x + y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    relative_radius: relative attention radius.
  """
  config: TransformerConfig
  relative_radius: Optional[int] = None

  @nn.compact
  def __call__(self,
               targets,
               encoded,
               relative_ids_dec=None,
               relative_ids_enc_dec=None,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      relative_ids_dec: relative ids used to calculate the decoder relative 
        position encodings.
      relative_ids_enc_dec: relative ids used to calculate the encoder-decoder 
        relative position encodings.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    cfg = self.config

    # Decoder block.
    assert targets.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(targets)
    x = SelfRelativeAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic,
        decode=cfg.decode,
        relative_radius=self.relative_radius,
        relative_bias=cfg.relative_bias)(x, relative_ids=relative_ids_dec, 
                                             mask=decoder_mask)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    y = MultiHeadDotProductRelativeAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic,
        relative_radius=self.relative_radius,
        relative_bias=cfg.relative_bias)(
            y, encoded, relative_ids_enc_dec, encoder_decoder_mask)

    y = nn.Dropout(rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic)
    y = y + x

    # MLP block.
    z = nn.LayerNorm(dtype=cfg.dtype)(y)
    z = MlpBlock(config=cfg)(z)

    return y + z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Any = None

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               encoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      encoder_mask: decoder self-attention mask.

    Returns:
      output of a transformer encoder.
    """
    cfg = self.config
    assert inputs.ndim == 2  # (batch, len)

    # Input Embedding
    if self.shared_embedding is None:
      input_embed = nn.Embed(
          num_embeddings=cfg.vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = self.shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)
    if cfg.sinusoidal:
      x = AddPositionEmbs(config=cfg, decode=False, name='posembed_input')(
          x, inputs_positions=inputs_positions)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)

    x = x.astype(cfg.dtype)
    
    relative_ids = None
    if cfg.relative_radius is not None:
      relative_ids = create_relative_ids(inputs.shape[1], cfg.relative_radius)
    
    # Input Encoder
    for lyr in range(cfg.num_layers):
      x = Encoder1DBlock(config=cfg, relative_radius=cfg.relative_radius,
                         name=f'encoderblock_{lyr}')(x, relative_ids, encoder_mask)
    encoded = nn.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)

    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Any = None

  @nn.compact
  def __call__(self,
               encoded,
               targets,
               targets_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      targets: target inputs.
      targets_positions: input subsequence positions for packed examples.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output of a transformer decoder.
    """
    cfg = self.config

    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    # Target Embedding
    if self.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=cfg.output_vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = self.shared_embedding

    y = targets.astype('int32')
    if not cfg.decode:
      y = shift_right(y)
    y = output_embed(y)
    if cfg.sinusoidal:
      y = AddPositionEmbs(config=cfg, decode=cfg.decode, name='posembed_output')(
          y, inputs_positions=targets_positions)
    y = nn.Dropout(rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic)

    y = y.astype(cfg.dtype)
    
    relative_ids_dec, relative_ids_enc_dec = None, None
    if cfg.relative_radius is not None:
        _, relative_ids_dec, relative_ids_enc_dec = create_relative_ids(encoded.shape[1], 
                                                                      cfg.relative_radius,
                                                                      targets.shape[1],
                                                                      None,
                                                                      cfg.enc2dec)
        if cfg.decode:
          is_initialized = self.has_variable('cache', 'cache_index')
          cache_index = self.variable('cache', 'cache_index',
                                      lambda: jnp.array(0, dtype=jnp.uint32))
          if is_initialized:
            idx = cache_index.value
            cache_index.value = idx + 1
            _, relative_ids_dec, relative_ids_enc_dec = create_relative_ids(encoded.shape[1], 
                                                                          cfg.relative_radius,
                                                                          targets.shape[1],
                                                                          cfg.max_len,
                                                                          cfg.enc2dec, idx)
    # Target-Input Decoder
    for lyr in range(cfg.num_layers):
      y = EncoderDecoder1DBlock(
          config=cfg, relative_radius=cfg.relative_radius, name=f'encoderdecoderblock_{lyr}')(
              y,
              encoded,
              relative_ids_dec=relative_ids_dec,
              relative_ids_enc_dec=relative_ids_enc_dec,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask)
    y = nn.LayerNorm(dtype=cfg.dtype, name='encoderdecoder_norm')(y)

    # Decoded Logits
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          cfg.output_vocab_size,
          dtype=cfg.dtype,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          name='logitdense')(y)
    return logits


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  def setup(self):
    cfg = self.config

    if cfg.share_embeddings:
      if cfg.output_vocab_size is not None:
        assert cfg.output_vocab_size == cfg.vocab_size, (
            "can't share embedding with different vocab sizes.")
      self.shared_embedding = nn.Embed(
          num_embeddings=cfg.vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      self.shared_embedding = None

    self.encoder = Encoder(config=cfg,
                           shared_embedding=self.shared_embedding)
    self.decoder = Decoder(config=cfg,
                           shared_embedding=self.shared_embedding)
                   
    if cfg.copy_decoder:
      self.final_layer_copy = nn.Dense(cfg.qkv_dim, kernel_init=cfg.kernel_init,
                                       bias_init=cfg.bias_init) # pe_input is the maximum input length
      self.final_layer_copy_weight = nn.Dense(1, kernel_init=cfg.kernel_init,
                                              bias_init=cfg.bias_init)
      # We want vocab_size -> vocab_size, but that might be too big.
      # So, we do a low-rank approximation, bringing it down to d_model first,
      # in case d_model < vocab_size:
      if cfg.qkv_dim < cfg.output_vocab_size:
        self.final_layer_copy2a = nn.Dense(cfg.qkv_dim, kernel_init=cfg.kernel_init,
                                               bias_init=cfg.bias_init)
        self.final_layer_copy2b = nn.Dense(cfg.output_vocab_size, kernel_init=cfg.kernel_init,
                                               bias_init=cfg.bias_init)
      else:
        self.final_layer_copy2 = nn.Dense(cfg.output_vocab_size, kernel_init=cfg.kernel_init,
                                              bias_init=cfg.bias_init)                       

  def encode(self,
             inputs,
             inputs_positions=None,
             inputs_segmentation=None):
    """Applies Transformer encoder-branch on the inputs.

    Args:
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      encoded feature array from the transformer encoder.
    """
    cfg = self.config
    
    # Make padding attention mask.
    encoder_mask = nn.make_attention_mask(
        inputs > 0, inputs > 0, dtype=cfg.dtype)
    # Add segmentation block-diagonal attention mask if using segmented data.
    if inputs_segmentation is not None:
      encoder_mask = nn.combine_masks(
          encoder_mask,
          nn.make_attention_mask(inputs_segmentation,
                                 inputs_segmentation,
                                 jnp.equal,
                                 dtype=cfg.dtype)
      )
    return self.encoder(
        inputs,
        inputs_positions=inputs_positions,
        encoder_mask=encoder_mask)

  def decode(self,
             encoded,
             inputs,  # only needed for masks
             targets,
             targets_positions=None,
             inputs_segmentation=None,
             targets_segmentation=None):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      inputs: input data (only needed for masking).
      targets: target data.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      logits array from transformer decoder.
    """
    cfg = self.config

    # Make padding attention masks.
    if cfg.decode:
      # for fast autoregressive decoding only a special encoder-decoder mask is used
      decoder_mask = None
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(targets) > 0, inputs > 0, dtype=cfg.dtype)
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(targets > 0, targets > 0, dtype=cfg.dtype),
          nn.make_causal_mask(targets, dtype=cfg.dtype))
      encoder_decoder_mask = nn.make_attention_mask(
          targets > 0, inputs > 0, dtype=cfg.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if inputs_segmentation is not None:
      decoder_mask = nn.combine_masks(
          decoder_mask,
          nn.make_attention_mask(targets_segmentation,
                                 targets_segmentation,
                                 jnp.equal,
                                 dtype=cfg.dtype))
      encoder_decoder_mask = nn.combine_masks(
          encoder_decoder_mask,
          nn.make_attention_mask(targets_segmentation,
                                 inputs_segmentation,
                                 jnp.equal,
                                 dtype=cfg.dtype))
    logits = self.decoder(
        encoded,
        targets,
        targets_positions=targets_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask)
    return logits.astype(self.config.dtype)

  def __call__(self,
               inputs,
               targets,
               inputs_positions=None,
               targets_positions=None,
               inputs_segmentation=None,
               targets_segmentation=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      targets: target data.
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      logits array from full transformer.
    """
    encoded = self.encode(inputs,
                          inputs_positions=inputs_positions,
                          inputs_segmentation=inputs_segmentation)
                          
    dec_output = self.decode(encoded,
                                 inputs,  # only used for masks
                                 targets,
                                 targets_positions=targets_positions,
                                 inputs_segmentation=inputs_segmentation,
                                 targets_segmentation=targets_segmentation)
    cfg = self.config
    if not cfg.copy_decoder:
      return dec_output
    else:
      final_output = nn.softmax(dec_output) # (batch_size, tar_seq_len, vocab_size)
      copy_output_query = self.final_layer_copy(dec_output)  # (batch_size, tar_seq_len, d_model)
      copy_output_weight = nn.sigmoid(self.final_layer_copy_weight(dec_output))
      copy_output = dot_product_relative_attention(
          copy_output_query, # (batch_size, tar_seq_len, d_model)
          encoded,         # (batch_size, inp_seq_len, d_model)
          jax.nn.one_hot(inputs, cfg.output_vocab_size)) # (batch_size, inp_seq_len, vocab_size)
      if cfg.qkv_dim < cfg.output_vocab_size:
          copy_output = nn.softmax(self.final_layer_copy2b(
              self.final_layer_copy2a(copy_output)))
      else:
          copy_output = nn.softmax(self.final_layer_copy2(copy_output))
            
      final_output = jnp.log(
          (1 - copy_output_weight) * final_output + copy_output_weight * copy_output)
      return final_output

