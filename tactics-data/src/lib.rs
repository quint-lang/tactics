/*
 * This source file is part of the quint-lang.org open source project
 *
 * Copyright (c) 2023 quint-lang
 * This program and the accompanying materials are made available under
 * the terms of the MIT License which is available at https://opensource.org/license/mit
 *
 * See https://quint-lang.org/tactics for more information
 */

use std::{fs::File, io::Read};

use csv::{ReaderBuilder, StringRecord};
use csv::ErrorKind::Seek;

use itertools::Itertools;

use ndarray::{
    iter::AxisChunksIter, Array, ArrayView, Axis, Dimension, IntoDimension, Ix, RemoveAxis, Zip,
};

use rand::{rngs::StdRng, Rng, SeedableRng, random};

use serde::de::DeserializeOwned;

/// Computes the correct shape for the stacked records of a dataset.
/// 该函数实现了一个功能，接受一个shape维度Dimension 和一个usize的rows，然后创建一个新的Dimension，维度为传入
/// 的 + 1,并将第一维度的值设置为rows， 之后将传入的Dimension的值赋予新的Dimension剩余的维度值
fn stacked_shape<D: Dimension>(rows: usize, shape: D) -> D::Larger {
    let mut new_shape = D::Larger::zeros(shape.ndim() + 1);
    new_shape[0] = rows;
    new_shape.slice_mut()[1..].clone_from_slice(shape.slice());

    new_shape
}

/// A collection of uniquely owned unlabeled records.
///
/// See also [*data*](index.html#data).
pub struct Dataset<D> {
    records: Array<f32, D>,
}

// Rust中不能直接在struct定义方法，一般是在impl中实现。这里impl<D: RemoveAxis> 表示这个方法只在 D 满足
// RemoveAxis trait 约束的情况下可用。这是一种对泛型参数的进一步约束，以确保只有满足特定条件的类型才能调用这个方法。
impl <D: RemoveAxis> Dataset<D> {
    /// Creates a new dataset from an [`Array`](ndarray::Array).
    ///
    /// # Arguments
    ///
    /// `records` - records to store into the dataset
    fn new(records: Array<f32, D>) -> Self {
        Self { records }
    }

    /// Returns a reference to the records.
    pub fn records(&self) -> &Array<f32, D> {
        &self.records
    }

    /// Returns the number of records stored in the dataset.
    pub fn len(&self) -> usize {
        self.records.len_of(Axis(0))
    }

    /// Checks whether the dataset is empty or not.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Constructs a K-Fold iterator from the dataset.
    ///
    /// The dataset is split *without shuffling* into k consecutive folds.
    ///
    /// Items returned by the [`KFold`] iterator are splits of the dataset. Each fold is used
    /// exactly once as a validation set, while the remaining k - 1 form the training set.
    ///
    /// Returned items are of the form `(Dataset, Dataset)`.
    ///
    /// # Arguments
    ///
    /// `k` - number of folds to perform.
    ///
    /// # Panics
    ///
    /// If `k < 2`.
    pub fn kfold(&self, k: usize) -> KFold<D> {
        KFold::new(self.records.view(), k)
    }

    /// Divides the dataset into batches of size `batch_size`.
    ///
    /// # Arguments
    ///
    /// `batch_size` - size of a single batch.
    pub fn batch(&self, batch_size: usize) -> Batch<D> {
        Batch::new(&self.records, batch_size)
    }

    /// Splits a dataset into non-overlapping new datasets of given lengths.
    ///
    /// # Arguments
    ///
    /// `lengths` -  lengths of splits to be produced.
    ///
    /// # Panics
    ///
    /// If the sum of the input lengths do not cover the whole dataset.
    pub fn split(self, lengths: &[usize]) -> Vec<Dataset<D>> {
        if self.len() != lengths.iter().sum::<usize>() {
            panic!("error: input lengths do not cover the whole dataset.");
        }

        let mut shape = self.records.raw_dim();
        let elems: Ix = shape.slice().iter().skip(1).product();
        let mut records = self.records.into_raw_vec();

        let mut datasets = Vec::with_capacity(lengths.len());
        for length in lengths {
            shape[0] = *length;

            datasets.push(Dataset::new(
                Array::from_shape_vec(shape.clone(),
                                      records.drain(..length * elems).collect()).unwrap()
            ));
        }
        datasets
    }

    /// Randomly shuffles the dataset.
    pub fn shuffle(&mut self) -> &mut Self {
        self.shuffle_with_seed(random())
    }

    /// Randomly shuffles the dataset.
    ///
    /// This version allows for a seed to be specified for results reproducibility.
    pub fn shuffle_with_seed(&mut self, seed: u64) -> &mut Self {
        let len = self.records.len_of(Axis(0));

        if len == 0 {
            return self;
        }

        let mut rng = StdRng::seed_from_u64(seed);
        for i in 0..len - 1 {
            // Since `iter.nth(pos)` consumes all the elements in `[0, pos]`
            // j will be in the interval `[pos + 1, len - 1]`, that contains `len - pos - 1`
            // elements
            let j = rng.gen_range(0..len - i - 1);

            let mut iter = self.records.outer_iter_mut();
            Zip::from(iter.nth(i).unwrap())
                .and(iter.nth(j).unwrap())
                .for_each(std::mem::swap);
        }

        self
    }
}

/// Configurable data loader.
pub struct DataLoader {
    r_builder: ReaderBuilder,
}

impl DataLoader {
    /// Specifies the columns where the labels are located.
    ///
    /// # Arguments
    ///
    /// `labels` - labels' columns indices.
    ///
    /// # Panics
    ///
    /// If the supplied labels are empty or if they contain duplicate columns.
    pub fn with_labels(self, labels: &[usize]) -> LabeledDataLoader {
        LabeledDataLoader::new(self, labels)
    }

    /// Configures the loader so that it parses the first row. To be used in the absence of an
    /// header row, as in most datasets the first row usually contains the columns' identifiers.
    pub fn without_headers(&mut self) -> &mut Self {
        self.r_builder.has_headers(false);

        self
    }

    /// Specifies the field delimiter character.
    ///
    /// # Arguments
    ///
    /// `delimiter` - delimiter character.
    pub fn with_delimiter(&mut self, delimiter: char) -> &mut Self {
        self.r_builder.delimiter(delimiter as u8);

        self
    }

    /// Builds a data collection by loading the content of the specified `.csv` file applying the
    /// previously supplied configuration.
    ///
    /// # Arguments
    ///
    /// * `src` - path of the source file.
    /// * `shape` - shape of a single record.
    ///
    /// # Panics
    ///
    /// In the case of errors during I/O or if `shape` generates an empty record.
    pub fn from_csv<S>(&mut self, src: &str, shape: S) -> Dataset<<S::Dim as Dimension>::Larger>
    where
        S: IntoDimension
    {
        self.from_reader_fn(File::open(src).unwrap(), shape, |r| r)
    }

    /// Builds a data collection by loading the content of the specified source reader applying the
    /// previously supplied configuration.
    ///
    /// # Arguments
    ///
    /// * `src` - reader from which to load the data.
    /// * `shape` - shape of a single record.
    ///
    /// # Panics
    ///
    /// In the event of a deserialization error or if `shape` would generate an empty record.
    pub fn from_reader<R, S>(&mut self, src: R, shape: S) -> Dataset<<S::Dim as Dimension>::Larger>
    where
        R: Read,
        S: IntoDimension
    {
        self.from_reader_fn(src, shape, |r| r)
    }

    /// Builds a data collection by loading the content of the specified `.csv` file applying the
    /// previously supplied configuration. Applies `fn` to each record.
    ///
    /// # Arguments
    ///
    /// * `src` - reader from which to load the data.
    /// * `shape` - shape of a single record.
    /// * `fn` - closure to be applied to each record.
    ///
    /// # Panics
    ///
    /// In the event of a deserialization error or if `shape` would generate an empty record.
    pub fn from_csv_fn<S, T, F>(
        &mut self,
        src: &str,
        shape: S,
        f: F,
    ) -> Dataset<<S::Dim as Dimension>::Larger>
    where
        S: IntoDimension,
        T: DeserializeOwned,
        F: Fn(T) -> Vec<f32>,
    {
        self.from_reader_fn(File::open(src).unwrap(), shape, f)
    }

    /// Builds a data collection by loading the content of the specified source reader applying the
    /// previously supplied configuration. Applies `fn` to each record.
    ///
    /// # Arguments
    ///
    /// * `src` - reader from which to load the data.
    /// * `shape` - shape of a single record.
    /// * `fn` - closure to be applied to each record.
    ///
    /// # Panics
    ///
    /// In the event of a deserialization error or if `shape` would generate an empty record.
    pub fn from_reader_fn<R, S, T, F>(
        &mut self,
        src: R,
        shape: S,
        f: F
    ) -> Dataset<<S::Dim as Dimension>::Larger>
    where
        R: Read,
        S: IntoDimension,
        T: DeserializeOwned,
        F: Fn(T) -> Vec<f32>,
    {
        let shape = shape.into_dimension();
        if shape.size() == 0 {
            panic!("error: cannot handle empty records.")
        }

        let mut records = Vec::new();
        let mut rows = 0;
        for record in self.r_builder.from_reader(src).deserialize() {
            let record = f(record.unwrap());
            records.extend(record);
            rows += 1;
        }

        Dataset::new(Array::from_shape_vec(stacked_shape(rows, shape), records).unwrap())
    }
}

impl Default for DataLoader {
    /// Creates a preconfigured data loader.
    ///
    /// The base configuration considers the first row of the source as an header, skipping it, and
    /// it uses `,` as the field delimiter.
    fn default() -> Self {
        Self {
            r_builder: ReaderBuilder::new(),
        }
    }
}

/// Configurable loader for labeled data.
pub struct LabeledDataLoader {
    r_bulder: ReaderBuilder,
    labels: Vec<usize>,
}

impl LabeledDataLoader {
    fn deserializer_record<T, U>(&self, record: StringRecord) -> (T, U)
    where
        T: DeserializeOwned,
        U: DeserializeOwned,
    {
        let mut input = StringRecord::new();
        let mut label = StringRecord::new();
        for (id, value) in record.iter().enumerate() {
            match self.labels.binary_search(&id) {
                Ok(_) => label.push_field(value),
                Err(_) => input.push_field(value),
            }
        }
        (
            input.deserialize(None).unwrap(),
            label.deserialize(None).unwrap(),
        )
    }

    fn new(builder: DataLoader, labels: &[usize]) -> Self {
        if labels.is_empty() {
            panic!("error: labels were not provided.");
        }

        let mut labels = labels.to_vec();
        labels.sort_unstable();
        if labels.windows(2).any(|w| w[0] == w[1]) {
            panic!("error: duplicated labels.");
        }

        Self {
            r_bulder: builder.r_builder,
            labels,
        }
    }

    /// Configures the loader so that it parses the first row. To be used in the absence of an
    /// header row, as in most datasets the first row usually contains the columns' identifiers.
    pub fn without_headers(&mut self) -> &mut Self {
        self.r_bulder.has_headers(false);

        self
    }

    /// Specifies the field delimiter character.
    ///
    /// # Arguments
    ///
    /// `delimiter` - delimiter character.
    pub fn with_delimiter(&mut self, delimiter: char) -> &mut Self {
        self.r_bulder.delimiter(delimiter as u8);

        self
    }

    /// Builds a labeled data collection by loading the content of the specified `.csv` file
    /// applying the previously supplied configuration.
    ///
    /// # Arguments
    ///
    /// * `src` - path of the source file.
    /// * `record_shape` - shape of a single record.
    /// * `label_shape` - shape of a single label.
    ///
    /// # Panics
    ///
    /// In the case of I/O errors or if `record_shape` generates an empty record or `label_shape`
    /// generates an empty label.
    pub fn from_csv<S1, S2>(
        &mut self,
        src: &str,
        record_shape: S1,
        label_shape: S2,
    ) -> LabeledDataset<<S1::Dim as Dimension>::Larger, <S2::Dim as Dimension>::Larger>
    where
        S1: IntoDimension,
        S2: IntoDimension,
    {
        self.from_reader_fn(File::open(src).unwrap(), record_shape, label_shape, |r| r)
    }

    /// Builds a data collection by loading the content of the specified source reader applying the
    /// previously supplied configuration.
    ///
    /// # Arguments
    ///
    /// * `src` - reader from which to load the data.
    /// * `record_shape` - shape of a single record.
    /// * `label_shape` - shape of a single label.
    ///
    /// # Panics
    ///
    /// In the event of a deserialization error or if `record_shape` generates an empty record or
    /// `label_shape` generates an empty label.
    pub fn from_reader<R, S1, S2>(
        &mut self,
        src: R,
        record_shape: S1,
        label_shape: S2,
    ) -> LabeledDataset<<S1::Dim as Dimension>::Larger, <S2::Dim as Dimension>::Larger>
    where
        R: Read,
        S1: IntoDimension,
        S2: IntoDimension,
    {
        self.from_reader_fn(src, record_shape, label_shape, |r| r)
    }

    /// Builds a data collection by loading the content of the specified `.csv` file applying the
    /// previously supplied configuration. Applies `fn` to each record and each label.
    ///
    /// # Arguments
    ///
    /// * `src` - reader from which to load the data.
    /// * `record_shape` - shape of a single record.
    /// * `label_shape` - shape of a single label.
    /// * `fn` - closure to be applied to records and labels.
    ///
    /// # Panics
    ///
    /// In the event of a deserialization error or if `record_shape` generates an empty record or
    /// `label_shape` generates an empty label.
    pub fn from_csv_fn<S1, S2, T, U, F>(
        &mut self,
        src: &str,
        record_shape: S1,
        label_shape: S2,
        f: F
    ) -> LabeledDataset<<S1::Dim as Dimension>::Larger, <S2::Dim as Dimension>::Larger>
    where
        S1: IntoDimension,
        S2: IntoDimension,
        T: DeserializeOwned,
        U: DeserializeOwned,
        F: Fn((T, U)) -> (Vec<f32>, Vec<f32>),
    {
        self.from_reader_fn(File::open(src).unwrap(), record_shape, label_shape, f)
    }

    /// Builds a data collection by loading the content of the specified source reader applying the
    /// previously supplied configuration. Applies `fn` to each record and each label.
    ///
    /// # Arguments
    ///
    /// * `src` - reader from which to load the data.
    /// * `record_shape` - shape of a single record.
    /// * `label_shape` - shape of a single label.
    /// * `fn` - closure to be applied to records and labels.
    ///
    /// # Panics
    ///
    /// In the event of a deserialization error if `record_shape` generates an empty record or
    /// `label_shape` generates an empty label.
    pub fn from_reader_fn<R, S1, S2, T, U, F>(
        &mut self,
        src: R,
        record_shape: S1,
        label_shape: S2,
        f: F,
    ) -> LabeledDataset<<S1::Dim as Dimension>::Larger, <S2::Dim as Dimension>::Larger>
    where
        R: Read,
        S1: IntoDimension,
        S2: IntoDimension,
        T: DeserializeOwned,
        U: DeserializeOwned,
        F: Fn((T, U)) -> (Vec<f32>, Vec<f32>),
    {
        let record_shape = record_shape.into_dimension();
        let label_shape = label_shape.into_dimension();
        if record_shape.size() == 0 || label_shape.size() == 0 {
            panic!("error: cannot handle empty records")
        }

        let mut records = Vec::new();
        let mut labels = Vec::new();
        let mut rows = 0;
        for record in self.r_bulder.from_reader(src).records() {
            let (record, label) = f(self.deserializer_record(record.unwrap()));
            records.extend(record);
            labels.extend(label);
            rows += 1;
        }

        LabeledDataset::new(
            Array::from_shape_vec(stacked_shape(rows, record_shape), records).unwrap(),
            Array::from_shape_vec(stacked_shape(rows, label_shape), labels).unwrap(),
        )
    }
}

/// A collection of uniquely owned *labeled* records.
///
/// `LabeledDataset` is generic on both the dimensionality of the records, specified by `D1` and
/// that of the labels, specified by `D2`. It's organized as a pair of tensors, one for the data and
/// one for the labels.
///
/// See also [*data*](index.html#data).
pub struct LabeledDataset<D1, D2> {
    records: Array<f32, D1>,
    labels: Array<f32, D2>,
}

impl<D1: RemoveAxis, D2: RemoveAxis> LabeledDataset<D1, D2> {
    /// Creates a new `LabeledDataset` from a pair of [`Array`]s.
    ///
    /// # Arguments
    ///
    /// * `records` - records to be stored.
    /// * `labels` - labels to be stored.
    fn new(records: Array<f32, D1>, labels: Array<f32, D2>) -> Self {
        Self { records, labels }
    }

    /// Returns a reference to the records.
    pub fn record(&self) -> &Array<f32, D1> {
        &self.records
    }

    /// Returns a reference to the labels.
    pub fn labels(&self) -> &Array<f32, D2> {
        &self.labels
    }

    /// Returns the number of records stored in the labeled dataset.
    pub fn len(&self) -> usize {
        self.records.len_of(Axis(0))
    }

    /// Check whether the labeled dataset is empty or not.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Constructs a K-Fold iterator from the labeled dataset.
    ///
    /// The dataset is split *without shuffling* into k consecutive folds.
    ///
    /// Items returned by the [`LabeledKFold`] iterator are splits of the dataset, both records and
    /// labels. Each fold is used exactly once as a validation set, while the remaining k - 1 form
    /// the training set.
    ///
    /// Returned items are of the form `(LabeledDataset, LabeledDataset)`.
    ///
    /// # Arguments
    ///
    /// `k` - number of folds to perform.
    ///
    /// # Panics
    ///
    /// If `k < 2`.
    pub fn kfold(&self, k: usize) -> LabeledKFold<D1, D2> {
        LabeledKFold::new(self.records.view(), self.labels.view(), k)
    }

    /// Divides the labeled dataset into batches of size `batch_size`.
    ///
    /// # Arguments
    ///
    /// `batch_size` - size of a single batch.
    pub fn batch(&self, size: usize) -> LabeledBatch<D1, D2> {
        LabeledBatch::new(&self.records, &self.labels, size)
    }

    /// Splits a labeled dataset into non-overlapping new datasets of given lengths.
    ///
    /// # Arguments
    ///
    /// `lengths` -  lengths of splits to be produced.
    ///
    /// # Panics
    ///
    /// If the sum of the input lengths do not cover the whole dataset.
    pub fn split(self, lengths: &[usize]) -> Vec<LabeledDataset<D1, D2>> {
        if self.len() != lengths.iter().sum::<usize>() {
            panic!("error: input lengths do not cover the whole dataset.");
        }

        let mut r_shape = self.records.raw_dim();
        let r_elems: Ix = r_shape.slice().iter().skip(1).product();
        let mut records = self.records.into_raw_vec();

        let mut l_shape = self.labels.raw_dim();
        let l_elems: Ix = l_shape.slice().iter().skip(1).product();
        let mut labels = self.labels.into_raw_vec();

        let mut datasets = Vec::with_capacity(lengths.len());
        for length in lengths {
            r_shape[0] = *length;
            l_shape[0] = *length;

            datasets.push(LabeledDataset::new(
                Array::from_shape_vec(r_shape.clone(), records.drain(..length * r_elems).collect())
                        .unwrap(),
                    Array::from_shape_vec(l_shape.clone(), labels.drain(..length * l_elems).collect())
                        .unwrap(),
            ));
        }

        datasets
    }

    /// Randomly shuffles the labeled dataset.
    pub fn shuffle(&mut self) -> &mut Self {
        self.shuffle_with_seed(random())
    }

    /// Randomly shuffles the labeled dataset.
    ///
    /// This version allows for a seed to be specified for results reproducibility.
    pub fn shuffle_with_seed(&mut self, seed: u64) -> &mut Self {
        let len = self.records.len_of(Axis(0));

        if len == 0 {
            return self;
        }

        let mut rng = StdRng::seed_from_u64(seed);
        for i in 0..len - 1 {
            // Since `iter.nth(pos)` consumes all the elements in `[0, pos]`
            // j will be in the interval `[pos + 1, len - 1]`, that contains `len - pos - 1`
            // elements
            let j = rng.gen_range(0..len - i - 1);

            let mut iter = self.records.outer_iter_mut();
            Zip::from(iter.nth(i).unwrap())
                .and(iter.nth(j).unwrap())
                .for_each(std::mem::swap);

            let mut iter = self.labels.outer_iter_mut();
            Zip::from(iter.nth(i).unwrap())
                .and(iter.nth(j).unwrap())
                .for_each(std::mem::swap);
        }

        self
    }
}

/// Iterator over batches of unlabeled data.
// 这里'a表示生命周期参数，生命周期参数通常用于确保引用的有效性。在这个情况下，'a 生命周期参数表示 Batch 结构体中的
// iter 字段引用的数据的生命周期必须至少与 'a 生命周期一样长，以确保引用的数据在使用期间有效。
pub struct Batch<'a, D> {
    iter: AxisChunksIter<'a, f32, D>,
}

impl<'a, D: RemoveAxis> Batch<'a, D> {
    /**
    *这个生命周期参数的作用是告诉 Rust 编译器，在函数内部，source 参数引用的数据在整个函数的执行过程中必须保持有效。这是 Rust 借用检查系统的一部分，用于确保引用的有效性和安全性。

    具体来说，'a 生命周期参数告诉编译器，source 参数引用的数据的生命周期不短于 'a 生命周期。这意味着在函数内部，source 参数引用的数据必须保持有效，直到 'a 生命周期结束。这可以防止在函数内部使用已经失效的引用，确保代码的正确性和安全性。

    在这个函数中，'a 生命周期参数允许你创建一个 Batch 结构体，并将其 iter 字段初始化为一个 AxisChunksIter 的实例，其中包含了 'a 生命周期，以确保在 Batch 实例中使用的 source 数据在 Batch 实例的生命周期内有效。
    */
    fn new(source: &'a Array<f32, D>, size: usize) -> Self {
        Self {
            iter: source.axis_chunks_iter(Axis(0), size),
        }
    }

    /// Drops the last incomplete batch, if the dataset size is not divisible by the batch size.
    pub fn drop_last(mut self) -> Self {
        let mut current = self.iter.clone();

        //Some() 是一个 Rust 中的 Option 枚举的成员之一，用于表示某个值存在的情况。
        // Option 通常用于处理可能为空（或不存在）的值，它有两个成员：Some(T) 和 None。
        // Some(T) 表示一个包含具体值 T 的情况，也就是某个值存在。
        // None 表示值不存在或为空的情况。
        // if let Some(next) = current.next() 和 if let Some(last) = current.last()
        // 这两行代码使用了 if let 表达式来匹配 current.next() 和 current.last() 的返回值是否为 Some。
        // 如果是 Some，则将其中的值绑定到 next 和 last 变量中。
        if let Some(next) = current.next() {
            if let Some(last) = current.last() {
                if next.len_of(Axis(0)) != last.len_of(Axis(0)) {
                    self.iter = self.iter.dropping_back(1);
                }
            }
        }

        self
    }
}

impl<'a, D: RemoveAxis> Iterator for Batch<'a, D> {
    type Item = <AxisChunksIter<'a, f32, D> as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

struct SetKFold<'a, D> {
    source: ArrayView<'a, f32, D>,
    step: usize,
    axis_len: usize,
}

impl<'a, D: RemoveAxis> SetKFold<'a, D> {
    pub fn new(source: ArrayView<'a, f32, D>, k: usize) -> Self {
        if k < 2 {
            panic!("error: folds must be > 2.");
        }

        let axis_len = source.len_of(Axis(0));
        debug_assert_ne!(axis_len, 0, "no record provided");

        Self {
            source,
            step: 1 + (axis_len - 1) / k,
            axis_len,
        }
    }

    pub fn compute_fold(&mut self, i: usize) -> (Array<f32, D>, Array<f32, D>) {
        let start = self.step * i;
        let stop = self.axis_len.min(start + self.step);

        let train_ids: Vec<usize> = (0..start).chain(stop..self.axis_len).collect();
        let test_ids: Vec<usize> = (start..stop).collect();

        (
            self.source.select(Axis(0), &train_ids),
            self.source.select(Axis(0), &test_ids),
        )
    }
}

/// K-Folds cross-validator on a labeled dataset.
pub struct LabeledKFold<'a, D1, D2> {
    records: SetKFold<'a, D1>,
    labels: SetKFold<'a, D2>,
    iteration: usize,
    k: usize,
}

impl<'a, D1, D2> LabeledKFold<'a, D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    pub fn new(records: ArrayView<'a, f32, D1>, labels: ArrayView<'a, f32, D2>, k: usize) -> Self {
        assert_eq!(records.len_of(Axis(0)), labels.len_of(Axis(0)));

        Self {
            records: SetKFold::new(records, k),
            labels: SetKFold::new(labels, k),
            iteration: 0,
            k,
        }
    }
}

impl<'a, D1, D2> Iterator for LabeledKFold<'a, D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    type Item = (LabeledDataset<D1, D2>, LabeledDataset<D1, D2>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iteration >= self.k {
            return None;
        }

        let (train_in, test_in) = self.records.compute_fold(self.iteration);
        let (train_out, test_out) = self.labels.compute_fold(self.iteration);
        self.iteration += 1;

        Some((
            LabeledDataset::new(train_in, train_out),
            LabeledDataset::new(test_in, test_out),
        ))
    }
}

/// Iterator over batches of labeled data.
pub struct LabeledBatch<'a, D1, D2> {
    records: Batch<'a, D1>,
    labels: Batch<'a, D2>,
}

impl<'a, D1: RemoveAxis, D2: RemoveAxis> LabeledBatch<'a, D1, D2> {
    fn new(records: &'a Array<f32, D1>, labels: &'a Array<f32, D2>, size: usize) -> Self {
        assert_eq!(records.len_of(Axis(0)), labels.len_of(Axis(0)));

        Self {
            records: Batch::new(records, size),
            labels: Batch::new(labels, size),
        }
    }

    /// Drops the last incomplete batch, if the dataset size is not divisible by the batch size.
    pub fn drop_last(mut self) -> Self {
        self.records = self.records.drop_last();
        self.labels = self.labels.drop_last();

        self
    }
}

impl<'a, D1: RemoveAxis, D2: RemoveAxis> Iterator for LabeledBatch<'a, D1, D2> {
    type Item = (
        <Batch<'a, D1> as Iterator>::Item,
        <Batch<'a, D2> as Iterator>::Item,
    );

    fn next(&mut self) -> Option<Self::Item> {
        match self.records.next() {
            None => None,
            Some(records) => Some((records, self.labels.next().unwrap())),
        }
    }
}

/// K-Folds cross-validator on a dataset.
pub struct KFold<'a, D> {
    records: SetKFold<'a, D>,
    iteration: usize,
    k: usize,
}

impl<'a, D: RemoveAxis> KFold<'a, D> {
    pub fn new(records: ArrayView<'a, f32, D>, k: usize) -> Self {
        Self {
            records: SetKFold::new(records, k),
            iteration: 0,
            k,
        }
    }
}

impl <'a, D: RemoveAxis> Iterator for KFold<'a, D> {
    type Item = (Dataset<D>, Dataset<D>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iteration >= self.k {
            return None;
        }

        let (records_in, records_out) = self.records.compute_fold(self.iteration);
        self.iteration += 1;

        Some((Dataset::new(records_in), Dataset::new(records_out)))
    }
}

#[cfg(test)]
mod test;