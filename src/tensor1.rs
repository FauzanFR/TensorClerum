use ndarray::{s, stack, Array1, Array2, ArrayViewMut1, Axis};

struct MemoryOrdo1 {
    x_max: usize,
    coords: Vec<(usize, usize, usize, usize)>,

}

impl MemoryOrdo1 {
    fn init() -> Self {
        Self { x_max: 0, coords: Vec::new() }
    }

    fn print(&self) {
        println!("x_max: {:?}", self.x_max);
        for (i, j, k, l) in &self.coords {
            println!("index: {} | {} ~ {}| layer: {}", i, j, k, l);
        }
    }

    fn make_layer(&self) -> Array1<f32> {
        Array1::<f32>::zeros(self.x_max)
    }

    fn get(&self, index: usize) -> Option<(usize, usize, usize)> {
        self.coords
            .iter()
            .find(|&&(idx, ..)| idx == index)
            .map(|&(_, x1, x2, z)| (x1, x2, z))
    }

    fn export(self) {

    }
}

pub struct arr1_to_tensor {
    memory: MemoryOrdo1,
    data_array1: Vec<Array1<f32>>,
    data_array1_fix: Vec<Array1<f32>>,
    data_array2_fix: Array2<f32>,
    true_index: Vec<usize>
}

impl arr1_to_tensor {

    pub fn new() -> Self{
        Self { memory: MemoryOrdo1::init(),
            data_array1: Vec::new(),
            data_array1_fix: Vec::new(),
            data_array2_fix: Array2::zeros((0,0)),
            true_index: Vec::new() }
    }

    pub fn push (& mut self, array : Array1<f32>) {
        self.data_array1.push(array);
    }

    pub fn init(self, array: Option<Vec<Array1<f32>>>) {
        match array {
            Some(arr) => {
                arr1_to_tensor::sorting(arr);
            }
            None => {
                arr1_to_tensor::sorting(self.data_array1);
            }
        }
    }

    fn sorting (array : Vec<Array1<f32>>) -> Self {
        let mut memory = MemoryOrdo1::init();
        let mut indexed: Vec<(usize, Array1<f32>)> = array
            .into_iter()
            .enumerate()
            .map(|(i,arr)|{
                memory.x_max = memory.x_max.max(arr.shape()[0]);
                (i, arr)
            })
            .collect();

        indexed.sort_by_key(|(_, v)| -(v.shape()[0] as isize));
        let (true_index, data_array1) = indexed.into_iter().unzip();

        Self {
            memory,
            data_array1,
            data_array1_fix: Vec::new(),
            data_array2_fix: Array2::zeros((1,1)),
            true_index
        }
    }

    pub fn count (&mut self) {
        let first_shape = self.data_array1[0].shape();
        let all_equal = self.data_array1.iter().all(|a| a.shape() == first_shape);
        if all_equal{self.stacking()} else {self.packing();}
    }

    fn stacking(&mut self){
        let mut layers: Vec<Array1<f32>> = Vec::new();
        let remaining: Vec<(usize, &Array1<f32>)> = self.data_array1.iter().enumerate().collect();
        
        for &(idx, arr) in &remaining{
            let mut layer = self.memory.make_layer();
            self.memory.coords.push((self.true_index[idx], 0, 0 + self.memory.x_max - 1, idx));
            layer.assign(arr);
            layers.push(layer);
        }
        self.data_array1_fix = layers;
        self.memory.print();
        self.data_array1 = Vec::new()
    }

    fn packing(&mut self){
        let mut layers: Vec<Array1<f32>> = Vec::new();
        let mut remaining: Vec<(usize, &Array1<f32>)> = self.data_array1.iter().enumerate().collect();
        let mut layer_id: usize = 0;

        while !remaining.is_empty() {
            let mut placed_idx = Vec::new();
            let mut layer = self.memory.make_layer();
            let mut free_space = self.memory.x_max;

            for &(idx, arr) in &remaining {
                let n = arr.dim();

                if free_space >= n {
                    layer.slice_mut(s![self.memory.x_max-free_space .. self.memory.x_max-free_space+n]).assign(arr);
                    self.memory.coords.push((self.true_index[idx], self.memory.x_max-free_space, 0 + self.memory.x_max-free_space+n-1, layer_id));
                    free_space -= n;
                    placed_idx.push(idx);
                }
            }

            layer_id += 1;
            layers.push(layer);
            remaining.retain(|(idx, _)| !placed_idx.contains(idx));
        }
        self.data_array1_fix = layers;
        self.memory.print();
        self.data_array1 = Vec::new()
    }

    pub fn to_tensor2 (&mut self){
        let views: Vec<_> = self.data_array1_fix.iter().map(|a| a.view()).collect();
        self.data_array2_fix = stack(Axis(0), &views).unwrap();
        self.data_array1_fix = Vec::new();
        self.true_index = Vec::new();
    }

    pub fn get(&mut self, index: usize) -> ArrayViewMut1<f32> {
        let (x1, x2, z) = self.memory
            .get(index)
            .expect(&format!("Invalid index {} in arr1_to_tensor::get", index));
        self.data_array2_fix.slice_mut(s![z, x1..=x2])
    }

    pub fn dim(&self, index: usize) -> usize {
        let (x1, x2, _) = self.memory
            .get(index)
            .expect(&format!("Invalid index {} in arr1_to_tensor::dim", index));
        x2 - x1 + 1
    }

    pub fn len(&self) -> usize {
        self.memory.coords.len()
    }

    pub fn export (self) -> (Array2<f32>, MemoryOrdo1) {
        (self.data_array2_fix, self.memory)
    }

    pub fn import (array : Array2<f32>, memory: MemoryOrdo1) -> Self {
        Self { memory: memory,
            data_array1: Vec::new(),
            data_array1_fix: Vec::new(),
            data_array2_fix: array,
            true_index: Vec::new() }
    }

}