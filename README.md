# Dinov2_finetune_model
使用 Hugging face 上的 Dinov2 水稻病害預訓練模型，對目前現有的水稻病害影像集進行微調。    
以多類別分類作為模型的分類任務。並使用 5-fold CV 檢視模型的穩定度。

Dinov2 的水稻病害預訓練模型網址    
https://huggingface.co/cvmil/dinov2-base_rice-leaf-disease-augmented_fft

水稻病害影像的資料集下載網址https://aidata.nchu.edu.tw/smarter/zh_Hant_TW/dataset/smarter_04_r14088_0_rice_20230118_img1_123456

隨機種子設定為 seed = 42    

微調訓練參數如下
1. Epoch = 15
2. patience = 5
3. optimizer = AdamW
4. learning rate 模型最後一層為 5e-5, 分類層為 1e-3
5. Warm-up ratio = 0.1
6. lr_scheduler_type: linear
7. criterion = CrossEntropyLoss()
8. batchsize = 16

每一個 fold 的 **precision, recall** 以及 **F1-score** 等指標已存入 **five_fold_per_class_metrics.csv**    
每一個 fold 的混淆矩陣已存入 **five_fold_confusion_matrix.csv**    



   
