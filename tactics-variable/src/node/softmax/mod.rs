/*
 * This source file is part of the quint-lang.org open source project
 *
 * Copyright (c) 2023 quint-lang
 * This program and the accompanying materials are made available under
 * the terms of the MIT License which is available at https://opensource.org/license/mit
 *
 * See https://quint-lang.org/tactics for more information
 */

use std::rc::Rc;

use ndarray::{Array, Axis, Dimension, Zip};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::Shared,
};

pub(crate) struct Softmax<D>
    where
        D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D>>,
    axis: Axis,
}

impl<D> Softmax<D>
    where
        D: Dimension,
{
    pub(crate) fn new(
        operand_data: Shared<Array<f32, D>>,
        data: Shared<Array<f32, D>>,
        axis: usize,
    ) -> Self {
        Self {
            operand_data,
            data,
            axis: Axis(axis),
        }
    }
}

impl<D> Forward for Softmax<D>
    where
        D: Dimension,
{
    fn forward(&self) {
        Zip::from(self.data.borrow_mut().lanes_mut(self.axis))
            .and(self.operand_data.borrow().lanes(self.axis))
            .for_each(|lane_v, lane_o| {
                let max = lane_o.fold(f32::MIN, |x, &y| x.max(y));
                let num = &lane_o.map(|&el| (el - max).exp());
                let den = num.sum();
                Zip::from(lane_v)
                    .and(num)
                    .for_each(|lane_v_el, &num_el| *lane_v_el = num_el / den);
            });
    }
}

pub(crate) struct SoftmaxBackward<D>
    where
        D: Dimension,
{
    operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
    data: Shared<Array<f32, D>>,
    gradient: Rc<Gradient<Array<f32, D>, D>>,
    axis: Axis,
}

impl<D> SoftmaxBackward<D>
    where
        D: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
        data: Shared<Array<f32, D>>,
        gradient: Rc<Gradient<Array<f32, D>, D>>,
        axis: usize,
    ) -> Self {
        Self {
            operand_gradient,
            data,
            gradient,
            axis: Axis(axis),
        }
    }
}

impl<D> Backward for SoftmaxBackward<D>
    where
        D: Dimension,
{
    fn backward(&self) {
        Zip::from(self.operand_gradient.borrow_mut().lanes_mut(self.axis))
            .and(self.gradient.borrow().lanes(self.axis))
            .and(self.data.borrow().lanes(self.axis))
            .for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let sum = Zip::from(grad_lane)
                    .and(data_lane)
                    .fold(0., |acc, &grad_el, &data_el| acc + grad_el * data_el);
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, &grad_el, &data_el| {
                        *op_grad_el += data_el * (grad_el - sum)
                    });
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
