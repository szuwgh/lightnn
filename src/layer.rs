use ndarray as nd;
use ndarray::{Array, Array1, Array2, Array3};
use std::ops::{Add, Mul};

use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    weights: Array1<Array2<f32>>,
    biases: Array1<Array2<f32>>,
}

impl Network {
    fn new(sizes: &[usize]) -> Network {
        Self {
            num_layers: sizes.len(),
            sizes: sizes.to_vec(),
            biases: Array::from_vec(
                sizes[1..]
                    .iter()
                    .map(|y| Array::random((*y, 1), Normal::<f32>::new(0., 1.).unwrap()))
                    .collect::<Vec<_>>(),
            ),
            weights: Array::from_vec(
                sizes
                    .iter()
                    .zip(sizes[1..].iter())
                    .map(|(x, y)| Array::random((*y, *x), Normal::<f32>::new(0., 1.).unwrap()))
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn feedforward(&self, mut a: Array2<f32>) -> Array2<f32> {
        // Return the output of the network if ``a`` is input.
        self.biases
            .iter()
            .zip(self.weights.iter())
            .for_each(|(b, w)| {
                a = sigmoid(w.dot(&a).add(b));
            });
        a
    }

    fn update_mini_batch(&self) {
        //   self.biases.iter().map(|b| b.shape())
    }

    fn backprop(&self) {}

    fn SGD() {}
}

#[inline]
fn sigmoid(z: Array2<f32>) -> Array2<f32> {
    z.mapv_into(|v| 1.0 / (1.0 + (-v).exp()))
}

#[inline]
fn sigmoid_prime(z: Array2<f32>) -> Array2<f32> {
    sigmoid(z.clone()).mul(sigmoid(z.clone()).mapv_into(|v| -v + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network() {
        //let xs = [2, 3, 1];
        //let a = Array::random((3, 2), Normal::<f32>::new(0., 1.).unwrap());
        let network = Network::new(&[2, 3, 1]);
        println!("{}", network.biases);
        println!("{}", network.weights);
    }
}
