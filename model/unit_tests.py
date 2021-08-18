""" Unit tests for functions used in the transformer model.
"""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import jax.numpy as jnp

from models import create_relative_ids


class UnitTest(tf.test.TestCase, parameterized.TestCase):

  def test_make_relative_ids_enc_only(self):
    enc_relative_ids =  create_relative_ids(in_length=6, relative_radius=3)

    expected = jnp.array([
        [3, 2, 1, 0, 0, 0],  #
        [4, 3, 2, 1, 0, 0],  #
        [5, 4, 3, 2, 1, 0],  #
        [6, 5, 4, 3, 2, 1],  #
        [6, 6, 5, 4, 3, 2],  #
        [6, 6, 6, 5, 4, 3],  #
    ])
    self.assertAllEqual(expected, enc_relative_ids)
    
  def test_make_relative_ids_enc_and_dec(self):
    enc_relative_ids, dec_relative_ids, dec2enc_relative_ids = create_relative_ids(
                                                    in_length=6, relative_radius=3, 
                                                    tar_length1=6)
    expected_enc = jnp.array([
        [3, 2, 1, 0, 0, 0],  #
        [4, 3, 2, 1, 0, 0],  #
        [5, 4, 3, 2, 1, 0],  #
        [6, 5, 4, 3, 2, 1],  #
        [6, 6, 5, 4, 3, 2],  #
        [6, 6, 6, 5, 4, 3],  #
    ])
    
    expected_dec = expected_enc
    
    expected_dec2enc = jnp.array([
        [3, 3, 3, 3, 3, 3],  #
        [3, 3, 3, 3, 3, 3],  #
        [3, 3, 3, 3, 3, 3],  #
        [3, 3, 3, 3, 3, 3],  #
        [3, 3, 3, 3, 3, 3],  #
        [3, 3, 3, 3, 3, 3],  #
    ])
    self.assertAllEqual(expected_enc, enc_relative_ids)
    self.assertAllEqual(expected_dec, dec_relative_ids)
    self.assertAllEqual(expected_dec2enc, dec2enc_relative_ids)
    
  def test_make_relative_ids_different_tar_lengths(self):
    enc_relative_ids, dec_relative_ids, dec2enc_relative_ids = create_relative_ids(
                                                    in_length=6, relative_radius=3, 
                                                    tar_length1=6, tar_length2=7)
    expected_enc = jnp.array([
        [3, 2, 1, 0, 0, 0],  #
        [4, 3, 2, 1, 0, 0],  #
        [5, 4, 3, 2, 1, 0],  #
        [6, 5, 4, 3, 2, 1],  #
        [6, 6, 5, 4, 3, 2],  #
        [6, 6, 6, 5, 4, 3],  #
    ])
    
    expected_dec = jnp.array([
        [3, 2, 1, 0, 0, 0, 0],  #
        [4, 3, 2, 1, 0, 0, 0],  #
        [5, 4, 3, 2, 1, 0, 0],  #
        [6, 5, 4, 3, 2, 1, 0],  #
        [6, 6, 5, 4, 3, 2, 1],  #
        [6, 6, 6, 5, 4, 3, 2],  #
    ])
    
    expected_dec2enc = jnp.array([
        [3, 3, 3, 3, 3, 3],  #
        [3, 3, 3, 3, 3, 3],  #
        [3, 3, 3, 3, 3, 3],  #
        [3, 3, 3, 3, 3, 3],  #
        [3, 3, 3, 3, 3, 3],  #
        [3, 3, 3, 3, 3, 3],  #
    ])
    self.assertAllEqual(expected_enc, enc_relative_ids)
    self.assertAllEqual(expected_dec, dec_relative_ids)
    self.assertAllEqual(expected_dec2enc, dec2enc_relative_ids)
    
  def test_make_relative_ids_dec2enc_ids_true(self):
    enc_relative_ids, dec_relative_ids, dec2enc_relative_ids = create_relative_ids(
                                                    in_length=6, relative_radius=3, 
                                                    tar_length1=6, tar_length2=7,
                                                    dec2enc_ids=True)
    expected_enc = jnp.array([
        [3, 2, 1, 0, 0, 0],  #
        [4, 3, 2, 1, 0, 0],  #
        [5, 4, 3, 2, 1, 0],  #
        [6, 5, 4, 3, 2, 1],  #
        [6, 6, 5, 4, 3, 2],  #
        [6, 6, 6, 5, 4, 3],  #
    ])
    
    expected_dec = jnp.array([
        [3, 2, 1, 0, 0, 0, 0],  #
        [4, 3, 2, 1, 0, 0, 0],  #
        [5, 4, 3, 2, 1, 0, 0],  #
        [6, 5, 4, 3, 2, 1, 0],  #
        [6, 6, 5, 4, 3, 2, 1],  #
        [6, 6, 6, 5, 4, 3, 2],  #
    ])
    
    expected_dec2enc = expected_enc
    self.assertAllEqual(expected_enc, enc_relative_ids)
    self.assertAllEqual(expected_dec, dec_relative_ids)
    self.assertAllEqual(expected_dec2enc, dec2enc_relative_ids)
    
  def test_make_relative_ids_cache_idx(self):
    enc_relative_ids, dec_relative_ids, dec2enc_relative_ids = create_relative_ids(
                                                    in_length=6, relative_radius=3, 
                                                    tar_length1=6, tar_length2=7,
                                                    dec2enc_ids=True, cache_idx=2)
    expected_enc = jnp.array([
        [3, 2, 1, 0, 0, 0],  #
        [4, 3, 2, 1, 0, 0],  #
        [5, 4, 3, 2, 1, 0],  #
        [6, 5, 4, 3, 2, 1],  #
        [6, 6, 5, 4, 3, 2],  #
        [6, 6, 6, 5, 4, 3],  #
    ])
    
    expected_dec = jnp.array([
        [5, 4, 3, 2, 1, 0, 0],  #
        [6, 5, 4, 3, 2, 1, 0],  #
        [6, 6, 5, 4, 3, 2, 1],  #
        [6, 6, 6, 5, 4, 3, 2],  #
        [6, 6, 6, 6, 5, 4, 3],  #
        [6, 6, 6, 6, 6, 5, 4],  #
    ])
    
    expected_dec2enc = jnp.array([
        [5, 4, 3, 2, 1, 0],  #
        [6, 5, 4, 3, 2, 1],  #
        [6, 6, 5, 4, 3, 2],  #
        [6, 6, 6, 5, 4, 3],  #
        [6, 6, 6, 6, 5, 4],  #
        [6, 6, 6, 6, 6, 5],  #
    ])
    self.assertAllEqual(expected_enc, enc_relative_ids)
    self.assertAllEqual(expected_dec, dec_relative_ids)
    self.assertAllEqual(expected_dec2enc, dec2enc_relative_ids)
    
  def test_make_relative_ids_no_padding(self):
    enc_relative_ids, dec_relative_ids, dec2enc_relative_ids = create_relative_ids(
                                                    in_length=6, relative_radius=7, 
                                                    tar_length1=6, tar_length2=7,
                                                    dec2enc_ids=True)
    expected_enc = jnp.array([
        [7, 6, 5, 4, 3, 2],  #
        [8, 7, 6, 5, 4, 3],  #
        [9, 8, 7, 6, 5, 4],  #
        [10, 9, 8, 7, 6, 5],  #
        [11, 10, 9, 8, 7, 6],  #
        [12, 11, 10, 9, 8, 7],  #
    ])
    
    expected_dec = jnp.array([
        [7, 6, 5, 4, 3, 2, 1],  #
        [8, 7, 6, 5, 4, 3, 2],  #
        [9, 8, 7, 6, 5, 4, 3],  #
        [10, 9, 8, 7, 6, 5, 4],  #
        [11, 10, 9, 8, 7, 6, 5],  #
        [12, 11, 10, 9, 8, 7, 6],  #
    ])
    
    expected_dec2enc = expected_enc
    self.assertAllEqual(expected_enc, enc_relative_ids)
    self.assertAllEqual(expected_dec, dec_relative_ids)
    self.assertAllEqual(expected_dec2enc, dec2enc_relative_ids)
    
    
if __name__ == '__main__':
  tf.test.main()
