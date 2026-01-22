# Dinov2_finetune_model
使用 Hugging face 上的 Dinov2 水稻病害預訓練模型，對目前現有的水稻病害影像集進行微調。    
以多類別分類作為模型的分類任務。並使用 5-fold CV 檢視模型的穩定度。
    
隨機種子設定為 seed = 42    
微調訓練參數如下
1. Epoch = 15
2. patience = 5
3. optimizer = AdamW
4. learning rate 模型最後一層為 5e-5, 分類層為 1e-3
5. Warm-up ratio = 0.1
6. criterion = CrossEntropyLoss()

每一個 fold 的 **precision, recall** 以及 **F1-score** 等指標已存入 **five_fold_per_class_metrics.csv**    
每一個 fold 的混淆矩陣已存入 **five_fold_confusion_matrix.csv**    



   
