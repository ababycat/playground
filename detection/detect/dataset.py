
class COCO_Detection_Dataset(dst.CocoDetection):
    def __init__(self, root, annFile, transform, method, **method_config):
        super().__init__(root, annFile, transform)
        self.parse_data_method = method
        self.method_config = method_config
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.table = {x['id']:idx for idx, x in enumerate(cats)}
        self.max_val = 4
        
    def __getitem__(self, index):
#         try:
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            print('no object in img(ID):%d, in index:%d' %(img_id, index))
            target = np.zeros((1, img.size[0], img.size[1]), dtype=np.uint8)
            if self.transform is not None:
                img, target = self.transform(img, target)
            out_label, max_val = self.parse_data_method([], nodata=True, **self.method_config)
            if max_val > self.max_val:
                print(max_val, ',')
            return img, out_label

        target = []
        class_ids = []
        for ann in anns:
            target.append(coco.annToMask(ann)*255)
            id_number = self.table[ann['category_id']]
            class_ids.append(id_number)

        target = np.stack(target, axis=0)
        class_ids = torch.from_numpy(np.stack(class_ids, axis=0).reshape(-1, 1))

        if self.transform is not None:
            img, target = self.transform(img, target)
        bbox = get_bbox_from_mask(target)
        bbox = torch.from_numpy(bbox)
        out_label, max_val = self.parse_data_method([class_ids, bbox, target], **self.method_config)
        if max_val > self.max_val:
            print(max_val)
        global error_id
        if error_id != 0:
            print('error in Image id:%d, index:%d, MORE OBJECTS IN GRID ERROR' %(img_id, index))
        return img, out_label
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class COCO_Key_Points(dst.CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform)

    def __getitem__(self, index):
        img = 0
        target = 0
        bbox = 0
        return img, [target, bbox]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

