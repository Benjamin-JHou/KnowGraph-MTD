# 生成Reprensentation

## 运行环境配置
# 安装依赖
# 


```bash
# cmx_kpgt workspace
conda env create -f env.yaml -n KPGT

```



## 准备数据集
### 分类/回归表格
1. `dataset-name.csv`: 包含两列：`smiles`,`cls/reg value`，第一列的col_name必须是`smiles`  
2. 新建文件夹并遵循路径规则： `datasets/dataset-name/dataset-name.csv`

### 描述符生成  
将代码中中`--dataset name`改成自定义数据集的名称
```bash
# 先切换到cmx_kpgt/scripts 作为工作目录
conda activate KPGT
python preprocess_downstream_dataset.py --data_path ../datasets/ --dataset name
```

## 生成Representation
将代码中中`--dataset name`改成自定义数据集的名称
```bash
# 先切换到cmx_kpgt/scripts 作为工作目录
python extract_features.py --config base --model_path ../pretrained/base/base.pth --data_path ../datasets/ --dataset name
```

## 多靶点协同有效的候选天然药物
## 第一次先跑小 epoch 测试
```bash
python ProjectionHead_MIL.py --data_path ../datasets --gene_dirs gene1 gene2 gene3 --compound_dir compounds --epochs 5 --final_topk 5
##  epoch100最终
conda activate KPGT
python ProjectionHead_MIL.py --data_path ../datasets --gene_dirs gene1 gene2 gene3 --compound_dir compounds --epochs 100 --final_topk 20 --aggregation geometric --out_dir mil_results

python ProjectionHead_MIL.py --data_path ../datasets --gene_dirs ATRIP CSF1 PDCD1 SRC STAT3 TLR7 TNF --compound_dir CMAUP --epochs 100 --final_topk 20 --aggregation geometric --out_dir mil_results
```

## 带IC50的候选天然药物
```bash
python IC50_aware_MIL_scoring.py --data_path ../datasets --gene_dirs gene1 gene2 gene3 --compound_dir compounds --aggregation geometric --alpha 1.0 --final_topk 20 --out_dir mil_results

python IC50_aware_MIL_scoring.py --data_path ../datasets --gene_dirs ATRIP CSF1 PDCD1 SRC STAT3 TLR7 TNF  --compound_dir Herb --aggregation geometric --alpha 1.0 --final_topk 20 --out_dir mil_results
```

## 合并成NC_all
conda activate KPGT
python merge_NC_memmap.py \
  --data_path ../datasets \
  --input_dirs Herb1 Herb2 Herb3 \
  --out_dir Herb_all
