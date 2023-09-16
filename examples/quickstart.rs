/*
 * This source file is part of the quint-lang.org open source project
 *
 * Copyright (c) 2023 quint-lang
 * This program and the accompanying materials are made available under
 * the terms of the MIT License which is available at https://opensource.org/license/mit
 *
 * See https://quint-lang.org/tactics for more information
 */

use tactics;

fn main() {
    let W_h = tactics::rand((20, 20)).requires_grad();
    let W_x = tactics::rand((20, 10)).requires_grad();
    let x = tactics::rand((1, 10));
    let h = tactics::rand((1, 20));

    let h2h = W_h.mm(h.t());
    let i2h = W_x.mm(x.t());

    let next_h = (h2h + i2h).tanh();
    let loss = next_h.sum();

    loss.forward();
    loss.backward(1.0);
}