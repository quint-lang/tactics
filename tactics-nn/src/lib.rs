/*
 * This source file is part of the quint-lang.org open source project
 *
 * Copyright (c) 2023 quint-lang
 * This program and the accompanying materials are made available under
 * the terms of the MIT License which is available at https://opensource.org/license/mit
 *
 * See https://quint-lang.org/tactics for more information
 */

use ndarray::{Ix1, Ix2, Ix3, Ix4, Ix5};

use tactics_core::{Convolution, MatMatMulT};

use tactics_variable::{PaddingMode, VarDiff};

pub mod init;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Applies a **linear transformation** to the incoming data.
///
/// ```text
/// ʏ = xAᵀ + b
/// ```
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Linear {
    pub weight: VarDiff<Ix2>,
    pub bias: VarDiff<Ix1>,
}

impl Linear {
    /// Creates a linear layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` – size of each input sample.
    ///
    /// * `out_features` – size of each output sample.
    ///
    /// The learnable weight of the layer is of shape `(out_features, in_features)`. The learnable
    /// bias of the layer is of shape `out_features`.
    ///
    /// The values for both the weight and bias are initialized from *U(-k, k)* where
    /// `k = (1. / in_features as f32).sqrt()`.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = tactics_variable::zeros((out_features, in_features)).requires_grad();
        let bias = tactics_variable::zeros(out_features).requires_grad();
        let k = (1. / (in_features as f32)).sqrt();
        init::uniform(&weight, -k, k);
        init::uniform(&bias, -k, k);

        Self { weight, bias }
    }

    /// Applies the linear transformation *y = xA^T + b* to the incoming data.
    ///
    /// # Arguments
    ///
    /// `data` - a variable of shape *(N, in_features)*, the output's shape will be
    /// *(N, out_features)*.
    pub fn forward<I>(&self, input: I) -> VarDiff<Ix2>
        where
            I: MatMatMulT<VarDiff<Ix2>>,
            I::Output: Into<VarDiff<Ix2>>,
    {
        input.mm_t(self.weight.clone()).into() + self.bias.clone()
    }
}

/// A **long short-term memory (LSTM)** cell.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[allow(clippy::upper_case_acronyms)]
pub struct LSTMCell {
    pub weight_ih: VarDiff<Ix2>,
    pub weight_hh: VarDiff<Ix2>,
    pub bias_ih: VarDiff<Ix1>,
    pub bias_hh: VarDiff<Ix1>,
}

impl LSTMCell {
    /// Creates a new LSTMCell.
    ///
    /// # Arguments
    ///
    /// * `input_size` - number of expected features in the input.
    ///
    /// * `hidden_size` - number of features in the hidden state.
    ///
    /// All the weight and biases are initialized from *U(-k, k)* where
    /// `k = (1. / hidden_size as f32).sqrt()`.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let (weight_ih_shape, weight_hh_shape, bias_shape) = {
            let xhidden_size = 4 * hidden_size;
            (
                (xhidden_size, input_size),
                (xhidden_size, hidden_size),
                xhidden_size,
            )
        };
        let weight_ih = tactics_variable::zeros(weight_ih_shape).requires_grad();
        let weight_hh = tactics_variable::zeros(weight_hh_shape).requires_grad();
        let bias_ih = tactics_variable::zeros(bias_shape).requires_grad();
        let bias_hh = tactics_variable::zeros(bias_shape).requires_grad();

        let k = 1. / (hidden_size as f32).sqrt();
        init::uniform(&weight_ih, -k, k);
        init::uniform(&weight_hh, -k, k);
        init::uniform(&bias_ih, -k, k);
        init::uniform(&bias_hh, -k, k);

        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    /// Computes a single **LSTM step**.
    ///
    /// # Arguments
    ///
    /// * `state` - a tuple of tensors, both of shape *(batch, hidden_size)*, containing the
    /// initial hidden state for each element in the batch and the initial cell's state for
    /// each element in the batch.
    ///
    /// * `input` - a variable containing the input features of shape *(batch, input_size)*.
    ///
    /// The **output** is a tuple of tensors made of the next hidden state for each element in
    /// the batch, of shape *(batch, hidden_size)* and the next cell's state for each element in
    /// the batch, of shape *(batch, hidden_size)*.
    pub fn forward<I>(
        &self,
        state: (VarDiff<Ix2>, VarDiff<Ix2>),
        input: I,
    ) -> (VarDiff<Ix2>, VarDiff<Ix2>)
        where
            I: MatMatMulT<VarDiff<Ix2>>,
            I::Output: Into<VarDiff<Ix2>>,
    {
        let (cell_state, hidden) = state;
        let gates = hidden.mm_t(self.weight_hh.clone())
            + self.bias_hh.clone()
            + input.mm_t(self.weight_ih.clone()).into()
            + self.bias_ih.clone();
        let gate_shape = {
            let (gates_shape_rows, gates_shape_cols) = gates.data().dim();
            (gates_shape_rows, gates_shape_cols / 4)
        };
        let chunked_gates = gates.chunks(gate_shape);
        let (input_gate, forget_gate, cell_state_gate, output_gate) = (
            chunked_gates[0].clone().sigmoid(),
            chunked_gates[1].clone().tanh(),
            chunked_gates[2].clone().sigmoid(),
            chunked_gates[3].clone().sigmoid(),
        );
        let new_cell_state = forget_gate * cell_state + (input_gate * cell_state_gate);
        let new_hidden = output_gate * new_cell_state.clone().tanh();

        (new_cell_state, new_hidden)
    }
}

/// A **gated recurrent unit (GRU)** cell.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[allow(clippy::upper_case_acronyms)]
pub struct GRUCell {
    pub weight_ih: VarDiff<Ix2>,
    pub weight_hh: VarDiff<Ix2>,
    pub bias_ih: VarDiff<Ix1>,
    pub bias_hh: VarDiff<Ix1>,
}

impl GRUCell {
    /// Creates a new GRUCell.
    ///
    /// # Arguments
    ///
    /// * `input_size` - number of expected features in the input.
    ///
    /// * `hidden_size` - number of features in the hidden state.
    ///
    /// All the weight and biases are initialized from *U(-k, k)* where
    /// `k = (1. / hidden_size as f32).sqrt()`.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let (weight_ih_shape, weight_hh_shape, bias_shape) = {
            let xhidden_size = 3 * hidden_size;
            (
                (xhidden_size, input_size),
                (xhidden_size, hidden_size),
                xhidden_size,
            )
        };
        let weight_ih = tactics_variable::zeros(weight_ih_shape).requires_grad();
        let weight_hh = tactics_variable::zeros(weight_hh_shape).requires_grad();
        let bias_ih = tactics_variable::zeros(bias_shape).requires_grad();
        let bias_hh = tactics_variable::zeros(bias_shape).requires_grad();

        let k = 1. / (hidden_size as f32).sqrt();
        init::uniform(&weight_ih, -k, k);
        init::uniform(&weight_hh, -k, k);
        init::uniform(&bias_ih, -k, k);
        init::uniform(&bias_hh, -k, k);

        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    /// Computes a single GRU step.
    ///
    /// * `hidden` - a variable of shape *(batch, hidden_size)*, containing the initial hidden state
    /// for each element in the batch.
    ///
    /// * `input` - a variable containing the input features of shape *(batch, input_size)*.
    ///
    /// The output is a variable made of the next hidden state for each element in
    /// the batch, of shape *(batch, hidden_size)*.
    pub fn forward<I>(&self, hidden: VarDiff<Ix2>, input: I) -> VarDiff<Ix2>
        where
            I: MatMatMulT<VarDiff<Ix2>>,
            I::Output: Into<VarDiff<Ix2>>,
    {
        let (igates, hgates) = {
            (
                input.mm_t(self.weight_ih.clone()).into() + self.bias_ih.clone(),
                hidden.clone().mm_t(self.weight_hh.clone()) + self.bias_hh.clone(),
            )
        };
        let gate_shape = {
            let (gates_shape_rows, gates_shape_cols) = hgates.data().dim();
            (gates_shape_rows, gates_shape_cols / 3)
        };
        let (chunked_igates, chunked_hgates) =
            (igates.chunks(gate_shape), hgates.chunks(gate_shape));

        let reset_gate = (chunked_hgates[0].clone() + chunked_igates[0].clone()).sigmoid();
        let input_gate = (chunked_hgates[1].clone() + chunked_igates[1].clone()).sigmoid();
        let new_gate =
            (chunked_igates[2].clone() + (chunked_hgates[2].clone() * reset_gate)).tanh();
        (hidden - new_gate.clone()) * input_gate + new_gate
    }
}

/// Applies a temporal convolution over an input signal composed of several input planes.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Conv1d<T>
    where
        T: PaddingMode<Ix3>,
{
    pub padding: usize,
    pub padding_mode: T,
    pub stride: usize,
    pub dilation: usize,
    pub weight: VarDiff<Ix3>,
    pub bias: VarDiff<Ix2>,
}

impl<T> Conv1d<T>
    where
        T: PaddingMode<Ix3>,
{
    /// Creates a new Conv1d.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - number of planes in the input signal.
    ///
    /// * `out_channels` - number of planes in the output signal.
    ///
    /// * `kernel_size` - size of the kernel, a number for this one-dimensional case.
    ///
    /// * `padding` - padding to be applied to the input, a number for this one-dimensional case.
    ///
    /// * `padding_mode` - padding mode, it can be: [`Zero`], [`Constant`], [`Reflective`] or
    /// [`Replicative`].
    ///
    /// * `stride` - stride of the convolution, a number for this one-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a number for this
    /// one-dimensional case.
    ///
    /// The weight and the bias of the layer are initialized from *U(-k, k)* where
    /// `k = (1. /(in_channels * kernel_size) as f32).sqrt()`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        padding_mode: T,
        stride: usize,
        dilation: usize,
    ) -> Self {
        let weight =
            tactics_variable::zeros((out_channels, in_channels, kernel_size)).requires_grad();
        let bias = tactics_variable::zeros((out_channels, 1)).requires_grad();

        let k = (1. / (in_channels * kernel_size) as f32).sqrt();
        init::uniform(&weight, -k, k);
        init::uniform(&bias, -k, k);

        Self {
            padding,
            padding_mode,
            stride,
            dilation,
            weight,
            bias,
        }
    }

    /// Computes a 1-dimensional convolution *(cross correlation)*.
    ///
    /// # Arguments
    ///
    /// `input` - signal to convolve.
    ///
    /// The **input** must be of shape *(N, Cin, L)*
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **L** is the **length** of the input
    ///
    /// The **kernel** must be of shape *(Cout, Cin, Lk)*
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Lk** is the **length** of the kernel
    ///
    /// The resulting output shape will be *(N, Cout, Lout)*
    pub fn forward<I>(&self, input: I) -> VarDiff<Ix3>
        where
            VarDiff<Ix3>: Convolution<I, Ix3>,
    {
        todo!()
    }
}

/// Applies a **spatial convolution** over an input signal composed of several input planes.
///
/// See also [`GroupedConv2d`].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Conv2d<T>
    where
        T: PaddingMode<Ix4>,
{
    pub padding: (usize, usize),
    pub padding_mode: T,
    pub stride: (usize, usize),
    pub dilation: (usize, usize),
    pub weight: VarDiff<Ix4>,
    pub bias: VarDiff<Ix3>,
}

impl<T> Conv2d<T>
    where
        T: PaddingMode<Ix4>,
{
    /// Creates a new Conv2d.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - number of planes in the input signal.
    ///
    /// * `out_channels` - number of planes in the output signal.
    ///
    /// * `kernel_size` - size of the kernel, a 2-tuple for this two-dimensional case.
    ///
    /// * `padding` - padding to be applied to the input, a 2-tuple for this two-dimensional case.
    ///
    /// * `padding_mode` - padding mode, it can be: [`Zero`], [`Constant`], [`Reflective`] or
    /// [`Replicative`].
    ///
    /// * `stride` - stride of the convolution, a 2-tuple for this two-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a 2-tuple for this
    /// two-dimensional case.
    ///
    /// The weight and the bias are initialized from *U(-k, k)* where
    /// `k = (1. /(in_channels * kernel_w * kernel_h) as f32).sqrt()`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        padding: (usize, usize),
        padding_mode: T,
        stride: (usize, usize),
        dilation: (usize, usize),
    ) -> Self {
        let (kernel_h, kernel_w) = kernel_size;
        let weight = tactics_variable::zeros((out_channels, in_channels, kernel_h, kernel_w))
            .requires_grad();
        let bias = tactics_variable::zeros((out_channels, 1, 1)).requires_grad();

        let k = (1. / (in_channels * kernel_h * kernel_w) as f32).sqrt();
        init::uniform(&weight, -k, k);
        init::uniform(&bias, -k, k);

        Self {
            padding,
            padding_mode,
            stride,
            dilation,
            weight,
            bias,
        }
    }

    /// Computes a 2-dimensional convolution *(cross correlation)*.
    ///
    /// # Arguments
    ///
    /// `input` - the signal to convolve.
    ///
    /// The **input** must be of shape *(N, Cin, H, W)*
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **H** is the **height** of the input
    /// * **W** is the **width** of the input
    ///
    /// The **kernel** must be of shape *(Cout, Cin, Hk, Wk)*
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Hk** is the **height** of the kernel
    /// * **Wk** is the **width** of the kernel
    ///
    /// The resulting output shape will be *(N, Cout, Hout, Wout)*
    pub fn forward<I>(&self, input: I) -> VarDiff<Ix4>
        where
            VarDiff<Ix4>: Convolution<I, Ix4>,
    {
        todo!()
    }
}

/// Applies a **volumetric convolution** over an input signal composed of several input planes.
///
/// See also [`GroupedConv3d`].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Conv3d<T>
    where
        T: PaddingMode<Ix5>,
{
    pub padding: (usize, usize, usize),
    pub padding_mode: T,
    pub stride: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub weight: VarDiff<Ix5>,
    pub bias: VarDiff<Ix4>,
}

impl<T> Conv3d<T>
    where
        T: PaddingMode<Ix5>,
{
    /// Creates a new Conv3d.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - number of planes in the input signal.
    ///
    /// * `out_channels` - number of planes in the output signal.
    ///
    /// * `kernel_size` - size of the kernel, a 3-tuple for this three-dimensional case.
    ///
    /// * `padding` - padding to be applied to the input, a 3-tuple for this three-dimensional case.
    ///
    /// * `padding_mode` - padding mode, it can be: [`Zero`], [`Constant`], [`Reflective`] or
    /// [`Replicative`].
    ///
    /// * `stride` - stride of the convolution, a 3-tuple for this three-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a 3-tuple for this
    /// three-dimensional case.
    ///
    /// The weight and the bias of the layer are initialized from *U(-k, k)* where
    /// `k = (1. /(in_channels * kernel_d * kernel_w * kernel_h) as f32).sqrt()`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        padding: (usize, usize, usize),
        padding_mode: T,
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Self {
        let (kernel_d, kernel_h, kernel_w) = kernel_size;
        let weight =
            tactics_variable::zeros((out_channels, in_channels, kernel_d, kernel_h, kernel_w))
                .requires_grad();
        let bias = tactics_variable::zeros((out_channels, 1, 1, 1)).requires_grad();

        let k = (1. / (in_channels * kernel_d * kernel_h * kernel_w) as f32).sqrt();
        init::uniform(&weight, -k, k);
        init::uniform(&bias, -k, k);

        Self {
            padding,
            padding_mode,
            stride,
            dilation,
            weight,
            bias,
        }
    }

    /// Computes a 3-dimensional convolution *(cross correlation)*.
    ///
    /// # Arguments
    ///
    /// `input` - signal to convolve.
    ///
    /// The **input** must be of shape *(N, Cin, D, H, W)*
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **D** is the **depth** of the input
    /// * **H** is the **height** of the input
    /// * **W** is the **width** of the input
    ///
    /// The **kernel** must be of shape *(Cout, Cin, Dk,  Hk, Wk)*
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Dk** is the **depth** of the kernel
    /// * **Hk** is the **height** of the kernel
    /// * **Wk** is the **width** of the kernel
    ///
    /// The resulting output shape will be *(N, Cout, Dout, Hout, Wout)*
    pub fn forward<I>(&self, input: I) -> VarDiff<Ix5>
        where
            VarDiff<Ix5>: Convolution<I, Ix5>,
            <VarDiff<Ix5> as Convolution<I, Ix5>>::Output: Into<VarDiff<Ix5>>,
    {
        todo!()
    }
}