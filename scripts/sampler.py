from torch.utils.data import IterableDataset
# class WarpSampler(object):
#     def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
#         self.result_queue = Queue(maxsize=n_workers * 10)
#         self.processors = []
#         for i in range(n_workers):
#             self.processors.append(
#                 Process(target=sample_function, args=(User,
#                                                       usernum,
#                                                       itemnum,
#                                                       batch_size,
#                                                       maxlen,
#                                                       self.result_queue,
#                                                       np.random.randint(2e9)
#                                                       )))
#             self.processors[-1].daemon = True
#             self.processors[-1].start()

#     def next_batch(self):
#         return self.result_queue.get()

#     def close(self):
#         for p in self.processors:
#             p.terminate()
#             p.join()
# class NoSampler(object):
#     def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
#         self.result_queue = Queue(maxsize=n_workers * 10)
#         self.processors = []
#         for i in range(n_workers):
#             self.processors.append(
#                 Process(target=sample_function, args=(User,
#                                                       usernum,
#                                                       itemnum,
#                                                       batch_size,
#                                                       maxlen,
#                                                       self.result_queue,
#                                                       np.random.randint(2e9)
#                                                       )))
#             self.processors[-1].daemon = True
#             self.processors[-1].start()

#     def next_batch(self):
#         return self.result_queue.get()

#     def close(self):
#         for p in self.processors:
#             p.terminate()
#             p.join()