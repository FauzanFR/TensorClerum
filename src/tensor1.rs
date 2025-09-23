use ndarray::{s, stack, Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
#[derive(Clone)]
struct Tensor1Metadata {
    x_max: usize,
    x_min: usize,
    coords: Vec<(usize, usize, usize, usize)>,
}

impl Tensor1Metadata {
    fn init() -> Self {
        Self { x_max: 0, x_min: usize::MAX, coords: Vec::new() }
    }

    fn print(&self) {
        println!("x_max: {:?}", self.x_max);
        println!("x_min: {:?}", self.x_min);
        for (i, j, k, l) in &self.coords {
            println!("index: {} | {} ~ {}| layer: {}", i, j, k, l);
        }
    }

    fn make_layer(&self) -> Array1<f32> {
        Array1::<f32>::zeros(self.x_max)
    }

    fn get_coords(&self, index: usize) -> Option<(usize, usize, usize)> {
        self.coords
            .iter()
            .find(|&&(idx, ..)| idx == index)
            .map(|&(_, x1, x2, z)| (x1, x2, z))
    }
}

pub struct PackedTensor1DStorage {
    data: Array2<f32>, 
    metadata:Tensor1Metadata
}

pub struct PackedTensor2DRef<'a> {
    pub data: &'a Array2<f32>,
    pub metadata: &'a Tensor1Metadata,
}

#[derive(Clone)]
pub struct PackedTensor1D {
    metadata: Tensor1Metadata,
    data_array1: Vec<Array1<f32>>,
    data_array2: Array2<f32>,
    true_index: Vec<usize>
}

impl PackedTensor1D {
    pub fn from_vec(array: Vec<Array1<f32>>) -> Self{
        Self { metadata: Tensor1Metadata::init(),
            data_array1: array,
            data_array2: Array2::zeros((0,0)),
            true_index: Vec::new() }
    }

    pub fn new() -> Self{
        Self { metadata: Tensor1Metadata::init(),
            data_array1: Vec::new(),
            data_array2: Array2::zeros((0,0)),
            true_index: Vec::new() }
    }

    pub fn push (& mut self, array : Array1<f32>) {
        self.data_array1.push(array);
    }

    pub fn process (&mut self) {
        let array = std::mem::take(&mut self.data_array1);
        (self.metadata, self.data_array1, self.data_array2, self.true_index) = PackedTensor1D::sorting(array);
        self.count();
        self.to_tensor();
    }

    fn sorting(array: Vec<Array1<f32>>) -> (Tensor1Metadata, Vec<Array1<f32>>, Array2<f32>, Vec<usize>) {
        let mut metadata = Tensor1Metadata::init();
        let mut indexed: Vec<(usize, Array1<f32>)> = array
            .into_iter()
            .enumerate()
            .map(|(i, arr)| {
                metadata.x_max = metadata.x_max.max(arr.dim());
                metadata.x_min = metadata.x_min.min(arr.dim());
                (i, arr)
            })
            .collect();

        indexed.sort_by_key(|(_, v)| -(v.shape()[0] as isize));
        let (true_index, data_array1) = indexed.into_iter().unzip();

        (
            metadata,
            data_array1,
            Array2::zeros((0,0)),
            true_index,
        )
    }

    fn count (&mut self) {
        let all_equal =(self.metadata.x_max / self.metadata.x_min) < 2;
        if all_equal{self.stacking()} else {self.packing();}
    }

    fn stacking(&mut self){
        let remaining: Vec<(usize, &Array1<f32>)> = self.data_array1.iter().enumerate().collect();
        let mut layers: Vec<Array1<f32>> = Vec::new();

        for &(idx, arr) in &remaining {
            let mut layer = self.metadata.make_layer();
            self.metadata.coords.push((
                self.true_index[idx],
                0, arr.dim() - 1,
                idx
            ));

            layer.slice_mut(s![0..arr.dim()]).assign(arr);
            layers.push(layer);
        }

        self.data_array1 = layers;
    }

    fn packing(&mut self){
        let mut layers: Vec<Array1<f32>> = Vec::new();
        let mut remaining: Vec<(usize, &Array1<f32>)> = self.data_array1.iter().enumerate().collect();
        let mut layer_id: usize = 0;

        while !remaining.is_empty() {
            let mut placed_idx = Vec::new();
            let mut layer = self.metadata.make_layer();
            let mut free_space = self.metadata.x_max;

            for &(idx, arr) in &remaining {
                let n = arr.dim();

                if free_space >= n {
                    layer.slice_mut(s![self.metadata.x_max-free_space .. self.metadata.x_max-free_space+n]).assign(arr);
                    self.metadata.coords.push((self.true_index[idx], self.metadata.x_max-free_space, 0 + self.metadata.x_max-free_space+n-1, layer_id));
                    free_space -= n;
                    placed_idx.push(idx);
                } 
            }

            layer_id += 1;
            layers.push(layer);
            remaining.retain(|(idx, _)| !placed_idx.contains(idx));
        }
        self.data_array1 = layers;
    }

    fn to_tensor (&mut self){
        let views: Vec<_> = self.data_array1.iter().map(|a| a.view()).collect();
        self.data_array2 = stack(Axis(0), &views).unwrap();
        self.data_array1 = Vec::new();
        self.true_index = Vec::new();
    }

    pub fn get(&self, index: usize) -> ArrayView1<f32> {
        let (x1, x2, z) = self.metadata
            .get_coords(index)
            .expect(&format!("Invalid index {} in PackedTensor1D::get", index));
        self.data_array2.slice(s![z, x1..=x2])
    }

    pub fn get_mut(&mut self, index: usize) -> ArrayViewMut1<f32> {
        let (x1, x2, z) = self.metadata
            .get_coords(index)
            .expect(&format!("Invalid index {} in PackedTensor1D::get_mut", index));
        self.data_array2.slice_mut(s![z, x1..=x2])
    }

    pub fn dim(&self, index: usize) -> usize {
        let (x1, x2, _) = self.metadata
            .get_coords(index)
            .expect(&format!("Invalid index {} in PackedTensor1D::dim", index));
        x2 - x1 + 1
    }

    pub fn len(&self) -> usize {
        self.metadata.coords.len()
    }

    pub fn print_coords (&self) {
        self.metadata.print();
    }

    pub fn as_view(&self) -> PackedTensor2DRef<'_> {
        PackedTensor2DRef {
            data: &self.data_array2,
            metadata: &self.metadata,
        }
    }

    pub fn export (&self) -> PackedTensor1DStorage {
        PackedTensor1DStorage{
            data: self.data_array2.clone(),
            metadata: self.metadata.clone()}
    }

    pub fn into_storage(self) -> PackedTensor1DStorage {
        PackedTensor1DStorage {
            data: self.data_array2,
            metadata: self.metadata,
        }
    }

    pub fn import (data:PackedTensor1DStorage) -> Self {
        Self {
            metadata: data.metadata,
            data_array1: Vec::new(),
            data_array2: data.data,
            true_index: Vec::new()
        }
    }

    pub fn iter(&self) -> PackedTensor1DIter<'_> {
        PackedTensor1DIter {
            packed_tensor: self,
            index: 0,
        }
    }

    pub fn iter_mut(&mut self) -> PackedTensor1DIterMut<'_> {
        PackedTensor1DIterMut {
            packed_tensor: self,
            index: 0,
        }
    }

    pub fn into_iter(self) -> PackedTensor1DIntoIter {
        PackedTensor1DIntoIter {
            packed_tensor: self,
            index: 0,
        }
    }
}

pub struct PackedTensor1DIter<'a> {
    packed_tensor: &'a PackedTensor1D,
    index: usize,
}

impl<'a> Iterator for PackedTensor1DIter<'a> {
    type Item = ArrayView1<'a, f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.packed_tensor.len() {
            let result = self.packed_tensor.get(self.index);
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

pub struct PackedTensor1DIterMut<'a> {
    packed_tensor: &'a mut PackedTensor1D,
    index: usize,
}

impl<'a> Iterator for PackedTensor1DIterMut<'a> {
    type Item = (usize, ArrayViewMut1<'a, f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.packed_tensor.len() {
            let index = self.index;
            let result = {
                let ptr = self.packed_tensor as *mut PackedTensor1D;
                unsafe {
                    (&mut *ptr).get_mut(index)
                }
            };
            self.index += 1;
            Some((index, result))
        } else {
            None
        }
    }
}

pub struct PackedTensor1DIntoIter {
    packed_tensor: PackedTensor1D,
    index: usize,
}

impl Iterator for PackedTensor1DIntoIter {
    type Item = Array1<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.packed_tensor.len() {
            let result = self.packed_tensor.get(self.index).to_owned();
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }
}