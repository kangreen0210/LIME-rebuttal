from lmms_eval.models.bunny_3b import Bunny_3B
model=Bunny_3B("/ML-A100/team/mm/zk/models/Bunny-v1_0-3B")

test=['hi are you ok']
result=model.generate_until(test)
print(result)