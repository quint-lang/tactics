/*
 * This source file is part of the quint-lang.org open source project
 *
 * Copyright (c) 2023 quint-lang
 * This program and the accompanying materials are made available under
 * the terms of the MIT License which is available at https://opensource.org/license/mit
 *
 * See https://quint-lang.org/tactics for more information
 */

use std::{
    cell::{Ref, RefCell, RefMut},
    collections::BTreeMap,
};

/// Id of an operation in the tape. The first component is the address of the struct and the second
/// is the size of the history at insertion. The former is unique, the latter enforces order.
// 下面是命名了一个结构体，其中包含一个元组类型的未命名字段，rust允许使用这种方式简写
// 下面是包含多个未命名字段的例子，如果想访问这些字段，可以通过模式匹配或者下标索引
// struct Test(i32, f64, bool);
//
// fn test()
// {
//     // 创建一个 MyTupleStruct 实例
//     let my_tuple = Test(42, 3.14, true);
//
//     // 使用模式匹配解构字段
//     let Test(first_field, second_field, third_field) = my_tuple;
//     // 使用索引访问字段
//     let first_field = my_tuple.0;   // 访问第一个字段，i32 类型
//     let second_field = my_tuple.1;  // 访问第二个字段，f64 类型
//     let third_field = my_tuple.2;   // 访问第三个字段，bool 类型
// }
#[derive(Copy, Clone, Eq)]
struct HistoryId((usize, usize));

impl HistoryId {
    /// Creates a new id from a pointer value and the instantaneous order of the op in the local
    /// tape.
    fn new(ptr: usize, order: usize) -> Self {
        Self((ptr, order))
    }
}

impl PartialEq for HistoryId {
    fn eq(&self, other: &Self) -> bool {
        let Self((lhs_ptr, _)) = self;
        let Self((rhs_ptr, _)) = other;

        lhs_ptr == rhs_ptr
    }
}

impl Ord for HistoryId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let Self((lhs_ptr, lhs_order)) = self;
        let Self((rhs_ptr, rhs_order)) = other;

        // Equal only if the pointers are the same.
        if lhs_ptr == rhs_ptr {
            return std::cmp::Ordering::Equal;
        }

        // But same ordering does not imply equality.
        if lhs_order == rhs_order {
            return std::cmp::Ordering::Less;
        }

        lhs_order.cmp(rhs_order)
    }
}

impl PartialOrd for HistoryId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
pub(crate) struct History<T>
    where
        T: Clone,
{
    path: BTreeMap<HistoryId, T>,
    buffer: RefCell<Vec<T>>,
}

impl<T> History<T>
    where
        T: Clone,
{
    /// Performs the merge between this history and another one.
    ///
    /// # Arguments
    ///
    /// `other` - other history.
    pub(crate) fn merge(&mut self, mut other: Self) {
        self.path.append(&mut other.path);
    }

    /// Appends a new computation to the history.
    ///
    /// # Arguments
    ///
    /// * `ptr` - address of the new node.
    ///
    /// * `op` - computation to append.
    pub(crate) fn insert(&mut self, ptr: usize, op: T) {
        let id = HistoryId::new(ptr, self.path.len());

        self.path.insert(id, op);
        self.buffer.borrow_mut().truncate(0);
    }

    /// Returns the length of the history.
    pub(crate) fn len(&self) -> usize {
        self.path.len()
    }

    /// Returns the length of the buffered history.
    pub(crate) fn buffer_len(&self) -> usize {
        self.buffer.borrow().len()
    }

    /// Returns the content of the tape in a vector.
    pub(crate) fn to_vec(&self) -> Vec<T> {
        self.path.values().cloned().collect()
    }

    /// Returns a reference to the buffer.
    pub(crate) fn buffer(&self) -> Ref<[T]> {
        Ref::map(self.buffer.borrow(), |buffer| &buffer[..])
    }

    /// Returns a mutable reference to the buffer.
    pub(crate) fn buffer_mut(&self) -> RefMut<Vec<T>> {
        self.buffer.borrow_mut()
    }
}

impl<T> Default for History<T>
    where
        T: Clone,
{
    fn default() -> Self {
        let path = BTreeMap::new();
        let buffer = RefCell::new(Vec::new());

        Self { path, buffer }
    }
}
