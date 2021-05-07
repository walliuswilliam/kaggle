import time

average_time = 0
for num in range(5):
  start = time.time()

  counter = 0
  for _ in range(1000000):
    counter += 1

  end = time.time()
  average_time += (end - start)
print(average_time/5)