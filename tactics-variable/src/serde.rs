/*
 * This source file is part of the quint-lang.org open source project
 *
 * Copyright (c) 2023 quint-lang
 * This program and the accompanying materials are made available under
 * the terms of the MIT License which is available at https://opensource.org/license/mit
 *
 * See https://quint-lang.org/tactics for more information
 */

use serde::{
    de::{Deserialize, Deserializer},
    ser::{Serialize, Serializer},
};

use ndarray::{Array, Dimension};

use super::{Var, VarDiff};

impl<D> Serialize for Var<D>
    where
        D: 'static + Dimension + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
    {
        self.data().serialize(serializer)
    }
}

impl<'d, D> Deserialize<'d> for Var<D>
    where
        D: 'static + Dimension + Deserialize<'d>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
        where
            De: Deserializer<'d>,
    {
        let data = Array::<f32, D>::deserialize(deserializer).unwrap();
        Ok(Self::leaf(data))
    }
}

impl<D> Serialize for VarDiff<D>
    where
        D: Dimension + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
    {
        self.data().serialize(serializer)
    }
}

impl<'d, D> Deserialize<'d> for VarDiff<D>
    where
        D: Dimension + Deserialize<'d>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
        where
            De: Deserializer<'d>,
    {
        let data = Array::<f32, D>::deserialize(deserializer).unwrap();
        Ok(Var::leaf(data).requires_grad())
    }
}
