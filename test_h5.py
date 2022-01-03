import h5py
## 取得h5py檔案的numpy
with h5py.File("/musicData/h5py_data/musdb18//ANiMAL - Clinic A/ANiMAL - Clinic A.h5","r") as f:
    for key in f.keys():
    	 #print(f[key], key, f[key].name, f[key].value) # 因爲這裏有group對象它是沒有value屬性的,故會異常。另外字符串讀出來是字節流，需要解碼成字符串。
        print(f[key], key, f[key].name)
        print(type(f[key][:]))
