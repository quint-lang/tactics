/*
 * This source file is part of the quint-lang.org open source project
 *
 * Copyright (c) 2023 quint-lang
 * This program and the accompanying materials are made available under
 * the terms of the MIT License which is available at https://opensource.org/license/mit
 *
 * See https://quint-lang.org/tactics for more information
 */

use ndarray::{ArrayView, ArrayViewMut, Dimension, RemoveAxis};

use super::SampleDim;

/// Padding mode.
pub trait PaddingMode<D>: Send + Sync + Copy
    where
        D: Dimension,
        D::Smaller: RemoveAxis,
{
    fn pad(
        &self,
        padded: &mut ArrayViewMut<f32, SampleDim<D>>,
        base: &ArrayView<f32, SampleDim<D>>,
        padding: SampleDim<D>,
    );
}
