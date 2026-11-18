# 0392 六位点预检修订独立复核

`/root/full80_review` 重新核对三份 V2 scope 和全部冻结源码。唯一执行代码变更是通过 `inspect.unwrap` 找到原生钩子函数定义，仍检查历史、实际和本次打包文件的三个一致 SHA。学校无推理预检复现了标准库装饰器路径并校验实际 hook 与两份历史 packed/打分源码哈希均通过。

两项 V1 作业在构建模型前退出，对应零 D4、零 NFE。静态 V1 未提交。下述三个 V2 scope 经独立批准，各 1 D4、4 NFE，一项一张卡，总并发上限两张。有效视频和模型预算保持三个视频、12 NFE。

- `readout-static-6` `3a5e3f623aec9a2173014b7eca4b058c3bf6b060d3aafcc84ac375499d8984e1`
- `readout-online-6` `4412b283996b7a8d1b61cd62a3e213c18f27d58a832dd5c9771da29f2624b10b`
- `Global-P64-6` `d34a7bb69145c5964b4334edf151ea464d5c5c9f1a340ac40ca5114358fdf4d8`
