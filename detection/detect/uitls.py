def get_bbox_from_mask(mask):
    """mask.shape=N*H*W or H*W"""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().type(torch.IntTensor).numpy()
    if len(mask.shape) == 3:
        out = []
        for i in range(mask.shape[0]):
            try:
                bbox = np.array(list(cv2.boundingRect(mask[i, :, :].astype(np.uint8))))
            except:
                print('get_bbox_from_mask_error')
                bbox = np.array([0, 0, 0, 0])
                continue
            out.append(bbox)
        return np.stack(out, axis=0)
    elif len(mask.shape) == 2:
        # return np.array(list(cv2.boundingRect(mask[i, :, :])))
        return np.array(list(cv2.boundingRect(mask[:, :]))).reshape(1, -1)
    else:
        raise 'error for mask.shape'