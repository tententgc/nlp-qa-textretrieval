# import time



# # print(model.args.fp16)
# context_text = open("inputs/context/1345136.txt").read()
# question = "ใครเป็นผู้ออกตราสารหนี้ภาคเอกชน ไร้ใบตราสาร"

# to_predict = [{
#         "context": context_text,
#         "qas": [{
#                 "question": question,
#                 "id": "0",
#             }],
#     }]
# st = time.time()
# # print()
# answers, probabilities = model.predict(to_predict)
# print(answers, time.time() - st)
# # print(*answers[0]['answer'])