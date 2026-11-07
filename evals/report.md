# MewHelp 离线评估报告（2026-11-07 17:34:29）

- 模型：`deepseek-chat` | 检索：向量+BM25 混合，rerank=on，top_k=4

## 意图分类

- 样本数：45，准确率：**86.7%**

| 类目 | Precision | Recall | F1 | 支持数 |
|---|---|---|---|---|
| delivery_time | 0.86 | 1.00 | 0.92 | 6 |
| greeting | 1.00 | 0.75 | 0.86 | 4 |
| irrelevant | 0.80 | 1.00 | 0.89 | 4 |
| order_issue | 1.00 | 0.67 | 0.80 | 6 |
| other | 0.80 | 1.00 | 0.89 | 4 |
| product_spec | 1.00 | 1.00 | 1.00 | 9 |
| refund | 1.00 | 1.00 | 1.00 | 3 |
| return_exchange | 0.50 | 1.00 | 0.67 | 3 |
| shipping_fee | 1.00 | 0.50 | 0.67 | 6 |

### 误分类明细

- g008: shipping_fee → return_exchange
- g009: shipping_fee → return_exchange
- g019: order_issue → delivery_time
- g040: order_issue → other
- g041: greeting → irrelevant
- g043: shipping_fee → return_exchange

## FAQ 检索

- 样本数：34（最终 top_k=4）
- Recall@1：**94.1%**
- Recall@3：**97.1%**
- Recall@5：**97.1%**
- MRR：**0.956**

### 未命中明细

- g038「快递到菜鸟驿站了我不在家怎么办？」→ ['## Q: 订单发货后可以修改地址吗？', '## Q: 逾期未收到货怎么处理？', '## Q: 收到的商品破损了怎么办？']
