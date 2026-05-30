# 必需接口检查清单

在跑验证前，确认以下方法已实现并正确接入：

- [ ] `handle_dataset`
- [ ] `init_model`
- [ ] `generate_model_visit`
- [ ] `generate_model_forward`
- [ ] `enable_kv_cache`

## 对齐检查

- [ ] `generate_model_visit` 与 `generate_model_forward` 遍历的层一致
- [ ] 遍历顺序一致
- [ ] 层间输入输出传递一致
- [ ] 检查 tokenizer 的 `pad_token` / `pad_token_id`：若为 `None`，已在适配器中重写 `_load_tokenizer` 并设置 `pad_token = eos_token`

## 注册检查

- [ ] `config/config.ini` 的 `[ModelAdapter]` 下已配置模型别名
- [ ] `config/config.ini` 的 `[ModelAdapterEntryPoints]` 下已配置入口
- [ ] 代码修改后已重新安装包
