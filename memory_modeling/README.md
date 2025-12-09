
# setup

modify your megatron to skip args validation in initialization.

```
diff --git a/megatron/training/initialize.py b/megatron/training/initialize.py
index e48d7380..90fc9a10 100644
--- a/megatron/training/initialize.py
+++ b/megatron/training/initialize.py
@@ -76,6 +76,11 @@ def initialize_megatron(
     if args.async_save and args.use_persistent_ckpt_worker:
         init_persistent_async_worker()
 
+    if allow_no_cuda and skip_mpu_initialization:
+        from .global_vars import set_args
+        set_args(args)
+        return
+
     if args.yaml_cfg is not None:
         args = validate_yaml(args, args_defaults)
     else:
```

add this megatron to PYTHONPATH

```
source env.sh PATH_TO_MEGATRON
```

# running

```
bash ./scripts/run_memory_modeling.py ./configs/360b-moe.sh
```


# no megatron running

set configs in ./scripts/run_autoperf_memory_modeling.sh

```
bash ./scripts/run_autoperf_memory_modeling.sh
```