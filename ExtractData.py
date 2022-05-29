import numpy as np
import scipy.sparse as sp
import pdb

class Dataset_aux(object):
    "'extract dataset from file'"

    def __init__(self, max_length, path, word_id_path):
        self.aux_word_id_dict = self.load_word_dict(path + word_id_path)
        print( "wordId_dict finished")
        self.aux_userReview_dict = self.load_reviews(max_length, len(self.aux_word_id_dict), path + "UserReviews.out")
        self.aux_itemReview_dict = self.load_reviews(max_length, len(self.aux_word_id_dict), path + "ItemReviews.out")
        print ("load reviews finished")
        self.num_users_aux, self.num_items_aux = len(self.aux_userReview_dict), len(self.aux_itemReview_dict)
        self.aux_trainMtrx = self.load_ratingFile_as_mtrx(path + "TrainInteraction.out")
        #self.valRatings = self.load_ratingFile_as_list(path + "ValidateInteraction.out")
        #self.testRatings = self.load_ratingFile_as_list(path + "TestInteraction.out")

    def load_word_dict(self, path):
        wordId_dict = {}

        with open(path, "r") as f:
            line = f.readline().replace("\n", "")
            while line != None and line != "":
                arr = line.split("\t")
                wordId_dict[arr[0]] = int(arr[1])
                line = f.readline().replace("\n", "")

        return wordId_dict

    def load_reviews(self, max_doc_length, padding_word_id, path):
        entity_review_dict = {}

        with open(path, "r") as f:
            line = f.readline().replace("\n", "")
            while line != None and line != "":
                review = []
                arr = line.split("\t")
                entity = int(arr[0])
                word_list = arr[1].split(" ")

                for i in range(len(word_list)):
                    if (word_list[i] == "" or word_list[i] == None or (not word_list[i] in self.aux_word_id_dict)):
                        continue
                    review.append(self.aux_word_id_dict.get(word_list[i]))
                    if (len(review) >= max_doc_length):
                        break
                if (len(review) < max_doc_length):
                    review = self.padding_word(max_doc_length, padding_word_id, review)
                entity_review_dict[entity] = review
                line = f.readline().replace("\n", "")
        return entity_review_dict

    def padding_word(self, max_size, max_word_idx, review):
        review.extend([max_word_idx]*(max_size - len(review)))
        return review

    def load_ratingFile_as_mtrx(self, file_path):
        mtrx = sp.dok_matrix((self.num_users_aux, self.num_items_aux), dtype=np.float32)
        with open(file_path, "r") as f:
            line = f.readline()
            line = line.strip()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mtrx[user, item] = rating
                line = f.readline()

        return mtrx

    def load_ratingFile_as_list(self, file_path):
        rateList = []

        with open(file_path, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                rate = float(arr[2])
                rateList.append([user, item, rate])
                line = f.readline()

        return rateList


class Dataset_tar(object):
    "'extract dataset from file'"

    def __init__(self, max_length, path, word_id_path):
        self.tar_word_id_dict = self.load_word_dict1(path + word_id_path)
        print( "wordId_dict finished")
        self.tar_userReview_dict = self.load_reviews1(max_length, len(self.tar_word_id_dict), path + "UserReviews.out")
        self.tar_itemReview_dict = self.load_reviews1(max_length, len(self.tar_word_id_dict), path + "ItemReviews.out")
        print ("load reviews finished")
        self.num_users_tar, self.num_items_tar = len(self.tar_userReview_dict), len(self.tar_itemReview_dict)
        self.tar_trainMtrx = self.load_ratingFile_as_mtrx1(path + "TrainInteraction.out")
        self.valRatings = self.load_ratingFile_as_list1(path + "ValidateInteraction.out")
        self.testRatings = self.load_ratingFile_as_list1(path + "TestInteraction.out")

    def load_word_dict1(self, path):
        wordId_dict = {}

        with open(path, "r") as f:
            line = f.readline().replace("\n", "")
            while line != None and line != "":
                arr = line.split("\t")
                wordId_dict[arr[0]] = int(arr[1])
                line = f.readline().replace("\n", "")

        return wordId_dict

    def load_reviews1(self, max_doc_length, padding_word_id, path):
        entity_review_dict = {}

        with open(path, "r") as f:
            line = f.readline().replace("\n", "")
            while line != None and line != "":
                review = []
                arr = line.split("\t")
                entity = int(arr[0])
                word_list = arr[1].split(" ")

                for i in range(len(word_list)):
                    if (word_list[i] == "" or word_list[i] == None or (not word_list[i] in self.tar_word_id_dict)):
                        continue
                    review.append(self.tar_word_id_dict.get(word_list[i]))
                    if (len(review) >= max_doc_length):
                        break
                if (len(review) < max_doc_length):
                    review = self.padding_word1(max_doc_length, padding_word_id, review)
                entity_review_dict[entity] = review
                line = f.readline().replace("\n", "")
        return entity_review_dict

    def padding_word1(self, max_size, max_word_idx, review):
        review.extend([max_word_idx]*(max_size - len(review)))
        return review

    def load_ratingFile_as_mtrx1(self, file_path):
        mtrx = sp.dok_matrix((self.num_users_tar, self.num_items_tar), dtype=np.float32)
        with open(file_path, "r") as f:
            line = f.readline()
            line = line.strip()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mtrx[user, item] = rating
                line = f.readline()

        return mtrx

    def load_ratingFile_as_list1(self, file_path):
        rateList = []

        with open(file_path, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                rate = float(arr[2])
                rateList.append([user, item, rate])
                line = f.readline()

        return rateList
