use ndarray::{s, stack, Array2, Array3, ArrayView2, ArrayViewMut2, Axis};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

impl Rect {
    fn area(&self) -> usize {
        self.w * self.h
    }
    
    fn contains(&self, other: &Rect) -> bool {
        self.x <= other.x &&
        self.y <= other.y &&
        self.x + self.w >= other.x + other.w &&
        self.y + self.h >= other.y + other.h
    }
    
    fn overlap(&self, other: &Rect) -> bool {
        self.x < other.x + other.w &&
        self.x + self.w > other.x &&
        self.y < other.y + other.h &&
        self.y + self.h > other.y
    }
}

struct MaxRectsBin {
    width: usize,
    height: usize,
    free_rects: Vec<Rect>,
    used_rects: Vec<Rect>,
}

impl MaxRectsBin {
    
    fn new(width: usize, height: usize) -> Self {
        let mut free_rects = Vec::new();
        free_rects.push(Rect { x: 0, y: 0, w: width, h: height });
        
        Self {
            width,
            height,
            free_rects,
            used_rects: Vec::new(),
        }
    }
    
    fn insert(&mut self, rect: &Rect) -> Option<(usize, usize)> {
        let mut best_node = None;
        let mut best_short_side_fit = std::usize::MAX;

        for (i, free_rect) in self.free_rects.iter().enumerate().rev() {
            if free_rect.w >= rect.w && free_rect.h >= rect.h {
                let leftover_h = free_rect.h - rect.h;
                let leftover_w = free_rect.w - rect.w;
                let short_side = leftover_h.min(leftover_w);
                
                if short_side < best_short_side_fit {
                    best_node = Some((free_rect.x, free_rect.y, i));
                    best_short_side_fit = short_side;
                }
            }
        }

        if let Some((x, y, free_rect_index)) = best_node {
            let placed_rect = Rect { x, y, w: rect.w, h: rect.h };
            self.used_rects.push(placed_rect);
            
            let free_rect = self.free_rects.remove(free_rect_index);
            self.split_free_rect(&free_rect, &placed_rect);
            self.merge_free_rects();
            
            Some((x, y))
        } else {
            None
        }
    }
    
    fn split_free_rect(&mut self, free_rect: &Rect, placed_rect: &Rect) {
        if placed_rect.y > free_rect.y {
            let new_rect = Rect {
                x: free_rect.x,
                y: free_rect.y,
                w: free_rect.w,
                h: placed_rect.y - free_rect.y,
            };
            if new_rect.w > 0 && new_rect.h > 0 {
                self.free_rects.push(new_rect);
            }
        }
        
        if placed_rect.y + placed_rect.h < free_rect.y + free_rect.h {
            let new_rect = Rect {
                x: free_rect.x,
                y: placed_rect.y + placed_rect.h,
                w: free_rect.w,
                h: free_rect.y + free_rect.h - (placed_rect.y + placed_rect.h),
            };
            if new_rect.w > 0 && new_rect.h > 0 {
                self.free_rects.push(new_rect);
            }
        }
        
        if placed_rect.x > free_rect.x {
            let new_rect = Rect {
                x: free_rect.x,
                y: free_rect.y,
                w: placed_rect.x - free_rect.x,
                h: free_rect.h,
            };
            if new_rect.w > 0 && new_rect.h > 0 {
                self.free_rects.push(new_rect);
            }
        }
        
        if placed_rect.x + placed_rect.w < free_rect.x + free_rect.w {
            let new_rect = Rect {
                x: placed_rect.x + placed_rect.w,
                y: free_rect.y,
                w: free_rect.x + free_rect.w - (placed_rect.x + placed_rect.w),
                h: free_rect.h,
            };
            if new_rect.w > 0 && new_rect.h > 0 {
                self.free_rects.push(new_rect);
            }
        }
    }
    
    fn merge_free_rects(&mut self) {
        let mut i = 0;
        while i < self.free_rects.len() {
            let mut j = i + 1;
            while j < self.free_rects.len() {
                if self.try_merge_rects(i, j) {
                    self.free_rects.remove(j);
                    i = 0;
                    break;
                }
                j += 1;
            }
            i += 1;
        }
    }
    
    fn try_merge_rects(&mut self, i: usize, j: usize) -> bool {
        let a = self.free_rects[i];
        let b = self.free_rects[j];
        
        if a.y == b.y && a.h == b.h {
            if a.x + a.w == b.x {
                self.free_rects[i] = Rect { x: a.x, y: a.y, w: a.w + b.w, h: a.h };
                return true;
            }
            if b.x + b.w == a.x {
                self.free_rects[i] = Rect { x: b.x, y: b.y, w: a.w + b.w, h: a.h };
                return true;
            }
        }
        
        if a.x == b.x && a.w == b.w {
            if a.y + a.h == b.y {
                self.free_rects[i] = Rect { x: a.x, y: a.y, w: a.w, h: a.h + b.h };
                return true;
            }
            if b.y + b.h == a.y {
                self.free_rects[i] = Rect { x: a.x, y: b.y, w: a.w, h: a.h + b.h };
                return true;
            }
        }
        
        false
    }
}

#[derive(Clone)]
struct Tensor2Metadata {
    x_max: usize,
    x_min: usize,
    y_max: usize,
    y_min: usize,
    coords: Vec<(usize, usize, usize, usize, usize, usize)>,
}

pub struct PackedTensor2DStorage {
    data: Array3<f32>, 
    metadata:Tensor2Metadata
}

pub struct PackedTensor2DRef<'a> {
    pub data: &'a Array3<f32>,
    pub metadata: &'a Tensor2Metadata,
}


impl Tensor2Metadata {
    fn init() -> Self {
        Self {
            x_max: 0,
            x_min: usize::MAX,
            y_max: 0,
            y_min: usize::MAX,
            coords: Vec::new(),
        }
    }

    fn get(&self, index: usize) -> Option<(usize, usize, usize, usize, usize)> {
        self.coords
            .iter()
            .find(|&&(idx, ..)| idx == index)
            .map(|&(_, x1, x2, y1, y2, z)| (x1, x2, y1, y2, z))
    }

    fn print(&self) {
        println!("x_max: {:?}", self.x_max);
        println!("x_min: {:?}", self.x_min);
        println!("y_max: {:?}", self.y_max);
        println!("y_min: {:?}", self.y_min);
        for (i, j, k, l, m, n) in &self.coords {
            println!("index: {} | {} ~ {}, {} ~ {} | layer: {}", i, j, k, l, m, n);
        }
    }

    fn make_layer(&self) -> Array2<f32> {
        Array2::<f32>::zeros((self.x_max, self.y_max))
    }

    fn make_max_rects_bin(&self) -> MaxRectsBin {
        MaxRectsBin::new(self.x_max, self.y_max)
    }
}

#[derive(Clone)]
pub struct PackedTensor2D {
    metadata: Tensor2Metadata,
    data_array2: Vec<Array2<f32>>,
    data_array3: Array3<f32>,
    true_index: Vec<usize>,
}

impl PackedTensor2D {

    pub fn from_vec(array: Vec<Array2<f32>>) -> Self{
        Self { metadata: Tensor2Metadata::init(),
            data_array2: array,
            data_array3: Array3::zeros((0,0,0)),
            true_index: Vec::new() }
    }

    pub fn new() -> Self{
        Self { metadata: Tensor2Metadata::init(),
            data_array2: Vec::new(),
            data_array3: Array3::zeros((0,0,0)),
            true_index: Vec::new() }
    }

    pub fn push (& mut self, array : Array2<f32>) {
        self.data_array2.push(array);
    }

    pub fn process (&mut self) {
        let array = std::mem::take(&mut self.data_array2);
        (self.metadata,self.data_array2, self.data_array3, self.true_index) = PackedTensor2D::sorting(array);
        self.count();
        self.to_tensor();
    }

    fn sorting(array: Vec<Array2<f32>>) -> (Tensor2Metadata, Vec<Array2<f32>>, Array3<f32>, Vec<usize>) {
        let mut metadata = Tensor2Metadata::init();
        let mut indexed: Vec<(usize, Array2<f32>)> = array
            .into_iter()
            .enumerate()
            .map(|(i, arr)| {
                let (x, y) = arr.dim();
                metadata.x_max = metadata.x_max.max(x);
                metadata.x_min = metadata.x_min.min(x);
                metadata.y_max = metadata.y_max.max(y);
                metadata.y_min = metadata.y_min.min(y);
                (i, arr)
            })
            .collect();

        indexed.sort_by_key(|(_, v)| -(v.shape()[0] as isize * v.shape()[1] as isize));
        let (true_index, data_array2) = indexed.into_iter().unzip();

        (
            metadata,
            data_array2,
            Array3::zeros((0,0,0)),
            true_index,
        )
    }

    fn count (&mut self) {
        let all_equal =
            (self.metadata.x_max / self.metadata.x_min) < 2 ||
            (self.metadata.y_max / self.metadata.y_min) < 2;
        if all_equal{self.stacking()} else {self.packing();}
    }
    
    fn stacking(&mut self){
        let mut layers: Vec<Array2<f32>> = Vec::new();
        let remaining: Vec<(usize, &Array2<f32>)> = self.data_array2.iter().enumerate().collect();

        for &(idx, arr) in &remaining {
            let (x, y) = arr.dim();
            let mut layer = self.metadata.make_layer();
            
            self.metadata.coords.push((
                self.true_index[idx],
                0, 0 + x - 1,
                0, 0 + y - 1,
                idx
            ));

            layer.slice_mut(s![0..x, 0..y]).assign(arr);
            layers.push(layer);
        }
        self.data_array2 = layers;
    }

    fn packing(&mut self){
        
        let mut remaining: Vec<(usize, &Array2<f32>)> = self.data_array2.iter().enumerate().collect();
        let mut layers: Vec<Array2<f32>> = Vec::new();
        let mut max_rects_bins: Vec<MaxRectsBin> = Vec::new();

        while !remaining.is_empty() {
            let mut placed_idx = Vec::new();

            for &(idx, arr) in &remaining {
                let (n1, n2) = arr.dim();
                let rect = Rect { x: 0, y: 0, w: n2, h: n1 };
                let mut placed = false;

                for (layer_id, (layer, bin)) in layers.iter_mut().zip(max_rects_bins.iter_mut()).enumerate() {
                    if let Some((x, y)) = bin.insert(&rect) {
                        layer.slice_mut(s![y..y + n1, x..x + n2]).assign(arr);
                        self.metadata.coords.push((self.true_index[idx], x, x + n2 - 1, y, y + n1 - 1, layer_id));
                        placed_idx.push(idx);
                        placed = true;
                        break;
                    }
                }

                if !placed {
                    let mut new_layer = self.metadata.make_layer();
                    let mut new_bin = self.metadata.make_max_rects_bin();
                    
                    if let Some((x, y)) = new_bin.insert(&rect) {
                        new_layer.slice_mut(s![y..y + n1, x..x + n2]).assign(arr);
                        self.metadata.coords.push((self.true_index[idx], x, x + n2 - 1, y, y + n1 - 1, layers.len()));
                        layers.push(new_layer);
                        max_rects_bins.push(new_bin);
                        placed_idx.push(idx);
                    } else {
                        panic!("Failed to insert into empty bin");
                    }
                }
            }

            remaining.retain(|(idx, _)| !placed_idx.contains(idx));
        }
        self.data_array2 = layers;
    }

    fn to_tensor (&mut self) {
        let views: Vec<_> = self.data_array2.iter().map(|a| a.view()).collect();
        self.data_array3 = stack(Axis(0), &views).unwrap();
        self.data_array2 = Vec::new();
        self.true_index = Vec::new();
    }

    pub fn get(&self, index: usize) -> ArrayView2<f32> {
        let (x1, x2, y1, y2, z) = self.metadata
            .get(index)
            .expect(&format!("Invalid index {} in PackedTensor2D::get", index));
        self.data_array3.slice(s![z, x1..=x2, y1..=y2])
    }

    pub fn get_mut(&mut self, index: usize) -> ArrayViewMut2<f32> {
        let (x1, x2, y1, y2, z) = self.metadata
            .get(index)
            .expect(&format!("Invalid index {} in PackedTensor2D::get_mut", index));
        self.data_array3.slice_mut(s![z, x1..=x2, y1..=y2])
    }

    pub fn dim (&self, index: usize) -> (usize, usize) {
        let (x1, x2, y1, y2, _) = self.metadata
            .get(index)
            .expect(&format!("Invalid index {} in PackedTensor2D::dim", index));
        (x2-x1+1, y2-y1+1)
    }
    
    pub fn len(&self) -> usize {
        self.metadata.coords.len()
    }

    pub fn print_coords (&self) {
        self.metadata.print();
    }

    pub fn as_view(&self) -> PackedTensor2DRef<'_> {
        PackedTensor2DRef {
            data: &self.data_array3,
            metadata: &self.metadata,
        }
    }

    pub fn export(&self) -> PackedTensor2DStorage {
        PackedTensor2DStorage {
            data: self.data_array3.clone(),
            metadata: self.metadata.clone(),
        }
    }

    pub fn into_storage(self) -> PackedTensor2DStorage {
        PackedTensor2DStorage {
            data: self.data_array3,
            metadata: self.metadata,
        }
    }

    pub fn import (data:PackedTensor2DStorage) -> Self {
        Self {
            metadata: data.metadata,
            data_array2: Vec::new(),
            data_array3: data.data,
            true_index: Vec::new()
        }
    }

    pub fn iter(&self) -> PackedTensor2DIter<'_> {
        PackedTensor2DIter {
            packed_tensor: self,
            index: 0,
        }
    }

    pub fn iter_mut(&mut self) -> PackedTensor2DIterMut<'_> {
        PackedTensor2DIterMut {
            packed_tensor: self,
            index: 0,
        }
    }

    pub fn into_iter(self) -> PackedTensor2DIntoIter {
        PackedTensor2DIntoIter {
            packed_tensor: self,
            index: 0,
        }
    }
}


pub struct PackedTensor2DIter<'a> {
    packed_tensor: &'a PackedTensor2D,
    index: usize,
}

impl<'a> Iterator for PackedTensor2DIter<'a> {
    type Item = ArrayView2<'a, f32>;

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

pub struct PackedTensor2DIterMut<'a> {
    packed_tensor: &'a mut PackedTensor2D,
    index: usize,
}

impl<'a> Iterator for PackedTensor2DIterMut<'a> {
    type Item = (usize, ArrayViewMut2<'a, f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.packed_tensor.len() {
            let index = self.index;
            let result = {
                let ptr = self.packed_tensor as *mut PackedTensor2D;
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

pub struct PackedTensor2DIntoIter {
    packed_tensor: PackedTensor2D,
    index: usize,
}

impl Iterator for PackedTensor2DIntoIter {
    type Item = Array2<f32>;

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