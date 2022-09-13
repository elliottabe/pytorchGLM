# README for "Joint coding of visual input and eye/head position in V1 of freely moving mice" dataset

## Opening h5 file
To open the data use code found in the github repo: Utils/io_dict_to_hdf5.py
Example: 

```
import Utils.io_dict_to_hdf5 as ioh5
data_all = ioh5.load(PATH_TO_FILE)
```


The dataset is organized as nested dictionaries:
 - Level 1: Date/animal
 - Level 2: Experimental condition - fm1 (freely moving) or hfwn (head-fixed white noise)
 - Level 3: Data variable

To view the shape of all variables: 
```
for key1 in data_all.keys():
    for key2 in data_all[key1].keys():
        for key3 in data_all[key1][key2].keys():
            print('_'.join([key1,key2,key3])+':',data_all[key1][key2][key3].shape)
```

Variables with the dictionary: 
- 070921_J553RT_fm1_model_active: (54354,)
- 070921_J553RT_fm1_model_eyerad: (54354,)
- 070921_J553RT_fm1_model_gz: (54354,)  
- 070921_J553RT_fm1_model_nsp: (54354, 128) 
- 070921_J553RT_fm1_model_phi: (54354,) 
- 070921_J553RT_fm1_model_pitch: (54354,) 
- 070921_J553RT_fm1_model_roll: (54354,) 
- 070921_J553RT_fm1_model_speed: (54354,) 
- 070921_J553RT_fm1_model_t: (54354,) 
- 070921_J553RT_fm1_model_th: (54354,) 
- 070921_J553RT_fm1_model_vid_sm: (54354, 30, 40) 
- 070921_J553RT_fm1_unit_nums: (128,) 
- 070921_J553RT_hfwn_model_eyerad: (18773,) 
- 070921_J553RT_hfwn_model_nsp: (18773, 128) 
- 070921_J553RT_hfwn_model_phi: (18773,) 
- 070921_J553RT_hfwn_model_t: (18773,) 
- 070921_J553RT_hfwn_model_th: (18773,) 
- 070921_J553RT_hfwn_model_vid_sm: (18773, 30, 40) 
- 070921_J553RT_hfwn_unit_nums: (128,) 
- 101521_J559NC_fm1_model_active: (72731,) 
- 101521_J559NC_fm1_model_eyerad: (72731,) 
- 101521_J559NC_fm1_model_gz: (72731,) 
- 101521_J559NC_fm1_model_nsp: (72731, 67) 
- 101521_J559NC_fm1_model_phi: (72731,) 
- 101521_J559NC_fm1_model_pitch: (72731,) 
- 101521_J559NC_fm1_model_roll: (72731,) 
- 101521_J559NC_fm1_model_speed: (72731,) 
- 101521_J559NC_fm1_model_t: (72731,) 
- 101521_J559NC_fm1_model_th: (72731,) 
- 101521_J559NC_fm1_model_vid_sm: (72731, 30, 40) 
- 101521_J559NC_fm1_unit_nums: (67,) 
- 101521_J559NC_hfwn_model_eyerad: (18993,) 
- 101521_J559NC_hfwn_model_nsp: (18993, 67) 
- 101521_J559NC_hfwn_model_phi: (18993,) 
- 101521_J559NC_hfwn_model_t: (18993,) 
- 101521_J559NC_hfwn_model_th: (18993,) 
- 101521_J559NC_hfwn_model_vid_sm: (18993, 30, 40) 
- 101521_J559NC_hfwn_unit_nums: (67,) 
- 102821_J570LT_fm1_model_active: (72729,) 
- 102821_J570LT_fm1_model_eyerad: (72729,) 
- 102821_J570LT_fm1_model_gz: (72729,) 
- 102821_J570LT_fm1_model_nsp: (72729, 56) 
- 102821_J570LT_fm1_model_phi: (72729,) 
- 102821_J570LT_fm1_model_pitch: (72729,) 
- 102821_J570LT_fm1_model_roll: (72729,) 
- 102821_J570LT_fm1_model_speed: (72729,) 
- 102821_J570LT_fm1_model_t: (72729,) 
- 102821_J570LT_fm1_model_th: (72729,) 
- 102821_J570LT_fm1_model_vid_sm: (72729, 30, 40) 
- 102821_J570LT_fm1_unit_nums: (56,) 
- 102821_J570LT_hfwn_model_eyerad: (18256,) 
- 102821_J570LT_hfwn_model_nsp: (18256, 56) 
- 102821_J570LT_hfwn_model_phi: (18256,) 
- 102821_J570LT_hfwn_model_t: (18256,) 
- 102821_J570LT_hfwn_model_th: (18256,) 
- 102821_J570LT_hfwn_model_vid_sm: (18256, 30, 40) 
- 102821_J570LT_hfwn_unit_nums: (56,) 
- 110421_J569LT_fm1_model_active: (56499,) 
- 110421_J569LT_fm1_model_eyerad: (56499,) 
- 110421_J569LT_fm1_model_gz: (56499,) 
- 110421_J569LT_fm1_model_nsp: (56499, 71) 
- 110421_J569LT_fm1_model_phi: (56499,) 
- 110421_J569LT_fm1_model_pitch: (56499,) 
- 110421_J569LT_fm1_model_roll: (56499,) 
- 110421_J569LT_fm1_model_speed: (56499,) 
- 110421_J569LT_fm1_model_t: (56499,) 
- 110421_J569LT_fm1_model_th: (56499,) 
- 110421_J569LT_fm1_model_vid_sm: (56499, 30, 40) 
- 110421_J569LT_fm1_unit_nums: (71,) 
- 110421_J569LT_hfwn_model_eyerad: (18340,) 
- 110421_J569LT_hfwn_model_nsp: (18340, 71) 
- 110421_J569LT_hfwn_model_phi: (18340,) 
- 110421_J569LT_hfwn_model_t: (18340,) 
- 110421_J569LT_hfwn_model_th: (18340,) 
- 110421_J569LT_hfwn_model_vid_sm: (18340, 30, 40) 
- 110421_J569LT_hfwn_unit_nums: (71,) 