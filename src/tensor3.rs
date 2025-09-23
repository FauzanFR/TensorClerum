use ndarray::{s, stack, Array3, Array4, ArrayView3, ArrayViewMut3, Axis};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
struct BoundingBox {
    min: [usize; 3],
    max: [usize; 3],
}

impl BoundingBox {
    fn new(min: [usize; 3], max: [usize; 3]) -> Self {
        Self { min, max }
    }

    fn intersects(&self, other: &BoundingBox) -> bool {
        self.min[0] <= other.max[0] && self.max[0] >= other.min[0] &&
        self.min[1] <= other.max[1] && self.max[1] >= other.min[1] &&
        self.min[2] <= other.max[2] && self.max[2] >= other.min[2]
    }

    fn contains(&self, other: &BoundingBox) -> bool {
        self.min[0] <= other.min[0] && self.max[0] >= other.max[0] &&
        self.min[1] <= other.min[1] && self.max[1] >= other.max[1] &&
        self.min[2] <= other.min[2] && self.max[2] >= other.max[2]
    }

    fn dimensions(&self) -> [usize; 3] {
        [
            self.max[0] - self.min[0] + 1,
            self.max[1] - self.min[1] + 1,
            self.max[2] - self.min[2] + 1,
        ]
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct OctreeNode {
    bounds: BoundingBox,
    children: Option<[Box<OctreeNode>; 8]>,
    occupied: bool,
    data_index: Option<usize>,
}

impl OctreeNode {
    fn new(bounds: BoundingBox) -> Self {
        Self {
            bounds,
            children: None,
            occupied: false,
            data_index: None,
        }
    }

    fn subdivide(&mut self) {
        let [min_x, min_y, min_z] = self.bounds.min;
        let [max_x, max_y, max_z] = self.bounds.max;
        
        let mid_x = (min_x + max_x) / 2;
        let mid_y = (min_y + max_y) / 2;
        let mid_z = (min_z + max_z) / 2;

        self.children = Some([
            Box::new(OctreeNode::new(BoundingBox::new([min_x, min_y, min_z], [mid_x, mid_y, mid_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([mid_x + 1, min_y, min_z], [max_x, mid_y, mid_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([min_x, mid_y + 1, min_z], [mid_x, max_y, mid_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([mid_x + 1, mid_y + 1, min_z], [max_x, max_y, mid_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([min_x, min_y, mid_z + 1], [mid_x, mid_y, max_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([mid_x + 1, min_y, mid_z + 1], [max_x, mid_y, max_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([min_x, mid_y + 1, mid_z + 1], [mid_x, max_y, max_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([mid_x + 1, mid_y + 1, mid_z + 1], [max_x, max_y, max_z]))),
        ]);
    }

    fn insert(&mut self, bounds: BoundingBox, data_index: usize) -> Option<BoundingBox> {
        if !self.bounds.contains(&bounds) {
            return None;
        }

        if self.occupied {
            return None;
        }

        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                if let Some(placement) = child.insert(bounds, data_index) {
                    return Some(placement);
                }
            }
            return None;
        }

        let node_dims = self.bounds.dimensions();
        let item_dims = bounds.dimensions();
        
        if node_dims[0] > item_dims[0] * 2 || 
           node_dims[1] > item_dims[1] * 2 || 
           node_dims[2] > item_dims[2] * 2 {
            self.subdivide();
            
            if let Some(children) = &mut self.children {
                for child in children.iter_mut() {
                    if let Some(placement) = child.insert(bounds, data_index) {
                        return Some(placement);
                    }
                }
            }
            return None;
        }
        
        self.occupied = true;
        self.data_index = Some(data_index);
        Some(self.bounds)
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct Tensor3Metadata {
    x_max: usize,
    x_min: usize,
    y_max: usize,
    y_min: usize,
    z_max: usize,
    z_min: usize,
    coords: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)>,
    octree: Option<OctreeNode>,
}

impl Tensor3Metadata {
    fn init(x_max: usize, y_max: usize, z_max: usize) -> Self {
        let root_bounds = BoundingBox::new([0, 0, 0], [x_max - 1, y_max - 1, z_max - 1]);
        
        Self {
            x_max,
            x_min: usize::MAX,
            y_max,
            y_min: usize::MAX,
            z_max,
            z_min: usize::MAX,
            coords: Vec::new(),
            octree: Some(OctreeNode::new(root_bounds)),
        }
    }

    fn print(&self) {
        println!("x_max: {:?}", self.x_max);
        println!("x_min: {:?}", self.x_min);
        println!("y_max: {:?}", self.y_max);
        println!("y_min: {:?}", self.y_min);
        println!("z_max: {:?}", self.z_max);
        println!("z_min: {:?}", self.z_min);
        for (i, j, k, l, m, n, o, p) in &self.coords {
            println!("index: {} | {} ~ {}, {} ~ {}, {} ~ {} | layer: {}", i, j, k, l, m, n, o, p);
        }
    }

    fn make_layer(&self) -> Array3<f32> {
        Array3::<f32>::zeros((self.x_max, self.y_max, self.z_max))
    }

    fn get(&self, index: usize) -> Option<(usize, usize, usize, usize, usize, usize, usize)> {
        self.coords
            .iter()
            .find(|&&(idx, ..)| idx == index)
            .map(|&(_, x1, x2, y1, y2, z1, z2, h)| (x1, x2, y1, y2, z1, z2, h))
    }

    fn insert_array(&mut self, index: usize, shape: [usize; 3]) -> Option<BoundingBox> {
        let bounds = BoundingBox::new([0, 0, 0], [shape[0] - 1, shape[1] - 1, shape[2] - 1]);
        
        if let Some(octree) = &mut self.octree {
            if let Some(placement) = octree.insert(bounds, index) {
                let [min_x, min_y, min_z] = placement.min;
                let [max_x, max_y, max_z] = placement.max;
                
                self.coords.push((
                    index,
                    min_x, max_x,
                    min_y, max_y,
                    min_z, max_z,
                    self.coords.len()
                ));
                
                return Some(placement);
            }
        }
        
        None
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PackedTensor3DStorage {
    data: Array4<f32>, 
    metadata: Tensor3Metadata
}

pub struct PackedTensor3DRef<'a> {
    data: &'a Array4<f32>, 
    metadata: &'a Tensor3Metadata
}

#[derive(Clone)]
pub struct PackedTensor3D {
    metadata: Tensor3Metadata,
    data_array3: Vec<Array3<f32>>,
    data_array4: Array4<f32>,
    true_index: Vec<usize>,
}

impl PackedTensor3D {
    pub fn from_vec(array: Vec<Array3<f32>>) -> Self {
        Self {
            metadata: Tensor3Metadata::init(2, 2, 2),
            data_array3: array,
            data_array4: Array4::zeros((0, 0, 0, 0)),
            true_index: Vec::new(),
        }
    }

    pub fn new() -> Self {
        Self {
            metadata: Tensor3Metadata::init(2, 2, 2),
            data_array3: Vec::new(),
            data_array4: Array4::zeros((0, 0, 0, 0)),
            true_index: Vec::new(),
        }
    }

    pub fn push(&mut self, array: Array3<f32>) {
        self.data_array3.push(array);
    }

    pub fn process(&mut self) {
        let array = std::mem::take(&mut self.data_array3);
        (self.metadata, self.data_array3, self.data_array4, self.true_index) = PackedTensor3D::sorting(array);
        self.count();
        self.to_tensor();
    }

    fn sorting(array: Vec<Array3<f32>>) -> (Tensor3Metadata, Vec<Array3<f32>>, Array4<f32>, Vec<usize>) {
        let mut max_dims = [0, 0, 0];
        let mut min_dims = [usize::MAX, usize::MAX, usize::MAX];
        
        for arr in &array {
            let dims = arr.dim();
            max_dims[0] = max_dims[0].max(dims.0);
            max_dims[1] = max_dims[1].max(dims.1);
            max_dims[2] = max_dims[2].max(dims.2);
            min_dims[0] = min_dims[0].min(dims.0);
            min_dims[1] = min_dims[1].min(dims.1);
            min_dims[2] = min_dims[2].min(dims.2);
        }

        let mut metadata = Tensor3Metadata::init(max_dims[0], max_dims[1], max_dims[2]);
        metadata.x_min = min_dims[0];
        metadata.y_min = min_dims[1];
        metadata.z_min = min_dims[2];
        
        let mut indexed: Vec<(usize, Array3<f32>)> = array.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| {
            let vol_a = a.1.dim().0 * a.1.dim().1 * a.1.dim().2;
            let vol_b = b.1.dim().0 * b.1.dim().1 * b.1.dim().2;
            vol_b.cmp(&vol_a)
        });

        let (true_index, data_array3) = indexed.into_iter().unzip();
        
        (
            metadata,
            data_array3,
            Array4::zeros((0, 0, 0, 0)),
            true_index,
        )
    }

    fn count(&mut self) {
        let all_equal =
            (self.metadata.x_max / self.metadata.x_min) < 2 ||
            (self.metadata.y_max / self.metadata.y_min) < 2 ||
            (self.metadata.z_max / self.metadata.z_min) < 2;
        
        if all_equal {
            self.stacking()
        } else {
            self.packing();
        }
    }

    fn stacking(&mut self) {
        let mut layers: Vec<Array3<f32>> = Vec::new();
        
        for (idx, arr) in self.data_array3.iter().enumerate() {
            let (x, y, z) = arr.dim();
            let mut layer = self.metadata.make_layer();
            
            self.metadata.coords.push((
                self.true_index[idx], 
                0, x - 1,
                0, y - 1,
                0, z - 1,
                idx
            ));
            
            layer.slice_mut(s![0..x, 0..y, 0..z]).assign(arr);
            layers.push(layer);
        }
        
        self.data_array3 = layers;
    }

    fn packing(&mut self) {
        let mut layers: Vec<Array3<f32>> = Vec::new();
        
        for (idx, arr) in self.data_array3.iter().enumerate() {
            let shape = arr.dim();
            let bounds = [shape.0, shape.1, shape.2];
            
            if let Some(placement) = self.metadata.insert_array(self.true_index[idx], bounds) {
                let layer_idx = self.metadata.coords.len() - 1;
                
                while layers.len() <= layer_idx {
                    layers.push(self.metadata.make_layer());
                }
                
                let [min_x, min_y, min_z] = placement.min;
                layers[layer_idx].slice_mut(s![
                    min_x..=placement.max[0],
                    min_y..=placement.max[1],
                    min_z..=placement.max[2]
                ]).assign(arr);
            } else {
                let mut layer = self.metadata.make_layer();
                layer.slice_mut(s![0..shape.0, 0..shape.1, 0..shape.2]).assign(arr);
                
                self.metadata.coords.push((
                    self.true_index[idx], 
                    0, shape.0 - 1,
                    0, shape.1 - 1,
                    0, shape.2 - 1,
                    layers.len()
                ));
                
                layers.push(layer);
            }
        }
        self.data_array3 = layers;
    }

    pub fn to_tensor(&mut self) {
        let views: Vec<_> = self.data_array3.iter().map(|a| a.view()).collect();
        self.data_array4 = stack(Axis(0), &views).unwrap();
        self.data_array3.clear();
        self.true_index.clear();
    }

    pub fn get(&self, index: usize) -> ArrayView3<f32> {
        let (x1, x2, y1, y2, z1, z2, h) = self.metadata
            .get(index)
            .unwrap_or_else(|| panic!("Invalid index {} in PackedTensor3D::get", index));
        self.data_array4.slice(s![h, x1..=x2, y1..=y2, z1..=z2])
    }

    pub fn get_mut(&mut self, index: usize) -> ArrayViewMut3<f32> {
        let (x1, x2, y1, y2, z1, z2, h) = self.metadata
            .get(index)
            .unwrap_or_else(|| panic!("Invalid index {} in PackedTensor3D::get_mut", index));
        self.data_array4.slice_mut(s![h, x1..=x2, y1..=y2, z1..=z2])
    }

    pub fn dim(&self, index: usize) -> (usize, usize, usize) {
        let (x1, x2, y1, y2, z1, z2, _) = self.metadata
            .get(index)
            .unwrap_or_else(|| panic!("Invalid index {} in PackedTensor3D::dim", index));
        (x2 - x1 + 1, y2 - y1 + 1, z2 - z1 + 1)
    }

    pub fn len(&self) -> usize {
        self.metadata.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    pub fn fill_all(&mut self, num:f32){
        self.data_array4.fill(num)
    }

    pub fn copy_and_fill(&self, num:f32) -> PackedTensor3D {
        let mut data = self.clone();
        data.data_array4.fill(num);
        data
    }

    pub fn print_coords(&self) {
        self.metadata.print();
    }

    pub fn as_view(&self) -> PackedTensor3DRef<'_> {
        PackedTensor3DRef {
            data: &self.data_array4,
            metadata: &self.metadata,
        }
    }

    pub fn export(&self) -> PackedTensor3DStorage {
        PackedTensor3DStorage {
            data: self.data_array4.clone(), 
            metadata: self.metadata.clone()
        }
    }

    pub fn into_storage(self) -> PackedTensor3DStorage {
        PackedTensor3DStorage {
            data: self.data_array4,
            metadata: self.metadata,
        }
    }

    pub fn import(data: PackedTensor3DStorage) -> Self {
        Self {
            metadata: data.metadata,
            data_array3: Vec::new(),
            data_array4: data.data,
            true_index: Vec::new(),
        }
    }

    pub fn iter(&self) -> PackedTensor3DIter<'_> {
        PackedTensor3DIter {
            packed_tensor: self,
            index: 0,
        }
    }

    pub fn iter_mut(&mut self) -> PackedTensor3DIterMut<'_> {
        PackedTensor3DIterMut {
            packed_tensor: self,
            index: 0,
        }
    }

    pub fn into_iter(self) -> PackedTensor3DIntoIter {
        PackedTensor3DIntoIter {
            packed_tensor: self,
            index: 0,
        }
    }
}

pub struct PackedTensor3DIter<'a> {
    packed_tensor: &'a PackedTensor3D,
    index: usize,
}

impl<'a> Iterator for PackedTensor3DIter<'a> {
    type Item = ArrayView3<'a, f32>;

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

pub struct PackedTensor3DIterMut<'a> {
    packed_tensor: &'a mut PackedTensor3D,
    index: usize,
}

impl<'a> Iterator for PackedTensor3DIterMut<'a> {
    type Item = (usize, ArrayViewMut3<'a, f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.packed_tensor.len() {
            let index = self.index;
            let result = {
                let ptr = self.packed_tensor as *mut PackedTensor3D;
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

pub struct PackedTensor3DIntoIter {
    packed_tensor: PackedTensor3D,
    index: usize,
}

impl Iterator for PackedTensor3DIntoIter {
    type Item = Array3<f32>;

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