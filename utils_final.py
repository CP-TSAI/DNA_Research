from IPython.display import clear_output
import numpy as np
import sys
import utils_final
import time
from operator import add
import matplotlib.pyplot as plt
from operator import add

# transform the className to a unit vector
def class2unitVector(className):

    # check the status of the string
    if not className:
        print("the className is EMPTY in class2unitVector()")
        time.sleep(5)


    className = className.upper()

    lst = [0] * 35

    if(className == "THANA"): 
        lst[0] = 1;
    elif(className == "HAN"): 
        lst[1] = 1;
    elif(className == "HAN_KO"): 
        lst[2] = 1;
    elif(className == "A"): 
        lst[3] = 1;
    elif(className == "AMI"): 
        lst[4] = 1;
    elif(className == "P"): 
        lst[5] = 1;
    elif(className == "PAI"): 
        lst[6] = 1;
    elif(className == "BS"): 
        lst[7] = 1;
    elif(className == "PUY"): 
        lst[8] = 1;
    elif(className == "C"): 
        lst[9] = 1;
    elif(className == "TSO"): 
        lst[10] = 1;
    elif(className == "T"): 
        lst[11] = 1;
    elif(className == "ATA"): 
        lst[12] = 1;
    elif(className == "B"): 
        lst[13] = 1;
    elif(className == "BUN"): 
        lst[14] = 1;
    elif(className == "S"): 
        lst[15] = 1;
    elif(className == "SAI"): 
        lst[16] = 1;
    elif(className == "L"): 
        lst[17] = 1;
    elif(className == "RUK"): 
        lst[18] = 1;
    elif(className == "Y"): 
        lst[19] = 1;
    elif(className == "TAO"): 
        lst[20] = 1;
    elif(className == "AUT"): 
        lst[21] = 1;
    elif(className == "EU"): 
        lst[22] = 1;
    elif(className == "GER"): 
        lst[23] = 1;
    elif(className == "FRENCH"): 
        lst[24] = 1;
    elif(className == "J"): 
        lst[25] = 1;
    elif(className == "UK"): 
        lst[26] = 1;
    elif(className == "HAK"): 
        lst[27] = 1;
    elif(className == "PIN"): 
        lst[28] = 1;
    elif(className == "JN"): 
        lst[29] = 1;
    elif(className == "JQ"): 
        lst[30] = 1;
    elif(className == "AY"): 
        lst[31] = 1;
    elif(className == "AP"): 
        lst[32] = 1;
    elif(className == "KC"): 
        lst[33] = 1;
    elif(className == "RCRS"): 
        lst[34] = 1;
    else:
        print("error name in class2unitVector(): ")
        print("className: ", className)
        lst[0] = 1;
        time.sleep(5)
    return lst

# transform the race to a unit vector
def race2unitVector(race):

    # check the status of the string
    if not race:
        print("the race is EMPTY in race2unitVector()")
        time.sleep(5)


    # race = race.upper()

    lst = [0] * 3

    if(race == 70): 
        lst[0] = 1;
    elif(race == 80): 
        lst[1] = 1;
    elif(race == 90): 
        lst[2] = 1;
    else:
        print("error name in race2unitVector(): ")
        print("race: ", race)
        lst[0] = 1;
        time.sleep(5)
    return lst

# if (fullName == EUL603) --> return EUL 
def fullName2Class(fullName):
    # print(fullName)
    # if (fullName[0:4] == "HAN0"): return "HAN0"


    fullName = fullName.upper();
    if (fullName == ".DS_STORE"):
        fullName = "KC993940";


    # fullName_processed = ""
    # for c in fullName:
    #     if (c != ' '):
    #         fullName_processed = fullName_processed + c
    # fullName = fullName_processed

    # differentiate "HAN" and "HAN_KO"
    if (fullName[0:3] == "HAN"):
        if (fullName[-2:] == "KO"):
            return "HAN_KO"
        else:
            return "HAN"


    classType = ""
    for c in fullName: # E, U, L, 6, 0, 3
        if (c.isalpha()): 
            classType = classType + c
        else: 
            break
    return classType

# 05/12
def class2index(className):

    if not className:
        print("the className is EMPTY in class2index()")
        time.sleep(20)

    className = className.upper(); 

    if(className == "THANA"): 
        return 0;
    elif(className == "HAN"): 
        return 1;
    elif(className == "HAN_KO"): 
        return 2;
    elif(className == "A"): 
        return 3;
    elif(className == "AMI"): 
        return 4;
    elif(className == "P"): 
        return 5;
    elif(className == "PAI"): 
        return 6;
    elif(className == "BS"): 
        return 7;
    elif(className == "PUY"): 
        return 8;
    elif(className == "C"): 
        return 9;
    elif(className == "TSO"): 
        return 10;
    elif(className == "T"): 
        return 11;
    elif(className == "ATA"): 
        return 12;
    elif(className == "B"): 
        return 13;
    elif(className == "BUN"): 
        return 14;
    elif(className == "S"): 
        return 15;
    elif(className == "SAI"): 
        return 16;
    elif(className == "L"): 
        return 17;
    elif(className == "RUK"): 
        return 18;
    elif(className == "Y"): 
        return 19;
    elif(className == "TAO"): 
        return 20;
    elif(className == "AUT"): 
        return 21;
    elif(className == "EU"): 
        return 22;
    elif(className == "GER"): 
        return 23;
    elif(className == "FRENCH"): 
        return 24;
    elif(className == "J"): 
        return 25;
    elif(className == "UK"): 
        return 26;
    elif(className == "HAK"): 
        return 27;
    elif(className == "PIN"): 
        return 28;
    elif(className == "JN"): 
        return 29;
    elif(className == "JQ"): 
        return 30;
    elif(className == "AY"): 
        return 31;
    elif(className == "AP"): 
        return 32;
    elif(className == "KC"): 
        return 33;
    elif(className == "RCRS"): 
        return 34;
    else:
        print("error name in class2index(): ")
        print("className: ", className)
        time.sleep(5)
        return 0;


def index2race(index):

    if(index == 0): 
        return 70;
    elif(index == 1): 
        return 70;
    elif(index == 2): 
        return 70;
    elif(index == 3): 
        return 70;
    elif(index == 4): 
        return 70;
    elif(index == 5): 
        return 70;
    elif(index == 6): 
        return 70;
    elif(index == 7): 
        return 70;
    elif(index == 8): 
        return 70;
    elif(index == 9): 
        return 70;
    elif(index == 10): 
        return 70;
    elif(index == 11): 
        return 70;
    elif(index == 12): 
        return 70;
    elif(index == 13): 
        return 70;
    elif(index == 14): 
        return 70;
    elif(index == 15): 
        return 70;
    elif(index == 16): 
        return 70;
    elif(index == 17): 
        return 70;
    elif(index == 18): 
        return 70;
    elif(index == 19): 
        return 70;
    elif(index == 20): 
        return 70;
    elif(index == 21): 
        return 80;
    elif(index == 22): 
        return 90;
    elif(index == 23): 
        return 80;
    elif(index == 24): 
        return 80;
    elif(index == 25): 
        return 70;
    elif(index == 26): 
        return 80;
    elif(index == 27): 
        return 70;
    elif(index == 28): 
        return 70;
    elif(index == 29): 
        return 90;
    elif(index == 30): 
        return 80;
    elif(index == 31): 
        return 80;
    elif(index == 32): 
        return 70;
    elif(index == 33): 
        return 70;
    elif(index == 34): 
        return 80;
    else:
        print("error index in index2race(): ")
        time.sleep(5)




def k_nearest_distance(df, K):
    original_class_array = []
    predicted_class_array = []

    original_race_array = []
    predicted_race_array = []

    for i in range(0, df.shape[0]):

        # visualize the process
        # clear_output(wait=True)
        print("Current Progress: ", np.round(i/df.shape[0] * 100, 2), "%")

        neighborClassType = []
        original_class = utils_final.class2index(utils_final.fullName2Class(df.loc[i][0]))
        original_race = utils_final.index2race(original_class)

        original_class_array.append(original_class)
        original_race_array.append(original_race)

        j = 1
        cnt = 1
        previous_distance = df.loc[i][2*j]
        
        while (cnt <= K and cnt < 10):

            # MARK: if running out data before we get enough k, errors could happen,
            # so need to have a size checker
            if (df.shape[1] <= 2*j+1): break

            if (previous_distance != df.loc[i][2*j]):
                previous_distance = df.loc[i][2*j]
                cnt += 1
            
            if (cnt <= K):
                neighborClassType.append(utils_final.class2index(utils_final.fullName2Class(df.loc[i][2*j-1])))
            j += 1

        
        # get the most frequent element

        predicted_class = utils_final.getMostFrequentNum(neighborClassType)
        predict_race = utils_final.index2race(predicted_class)

        predicted_class_array.append(predicted_class)
        predicted_race_array.append(predict_race)

        # print("neighborClassType: ", neighborClassType)
        # time.sleep(10)

    return original_class_array, predicted_class_array, original_race_array, predicted_race_array

# get K neighbors from df (DataFrame)
###Wrong!!!

def k_nearest_neighbor(df, K):
    original_race_array = []
    predicted_race_array = []
    for i in range(0, df.shape[0]):
        # visualize the process
        clear_output(wait=True)
        print("Current Progress: ", np.round(i/df.shape[0] * 100, 2), "%")
        # print(i)
        
        neighborRaceType = []
        for j in range(0, 2*K+1):
            if (j == 0): # object itself
                original_race = utils_final.index2race(utils_final.class2index(utils_final.fullName2Class(df.loc[i][j])))
                original_race_array.append(original_race)
            elif (j % 2 == 1): # neighbors
                # print("j: ", j)
                # print("df.loc[i][j]: ", df.loc[i][j])
                neighborRaceType.append(utils_final.index2race(utils_final.class2index(utils_final.fullName2Class(df.loc[i][j]))))   
        
        # get the most frequent element
        predicted_race = utils_final.getMostFrequentNum(neighborRaceType)
        predicted_race_array.append(predicted_race)
    return original_race_array, predicted_race_array


# get the most frequent element in a list
def getMostFrequentNum(lst):
    return max(set(lst), key=lst.count)


################################################
# calculate the accuracy of list1 and list2
# list1 = [1, 2, 2, 3, 2]   ground truth
# list2 = [1, 3, 2, 3, 1]   prediction
def getAccuracyOfTwoLists(lst1, lst2):
    if (len(lst1) != len(lst2)):
        print("ERROR!! Lengths of lists are different.")

    accuracy = 0
    for i in range(0, len(lst1)):
        if (lst1[i] == lst2[i]):
            accuracy = accuracy + 1

    if (len(lst1) == 0):
        print("LENGTH is 0");
        return 0;

    accuracy = accuracy / len(lst1)
    return accuracy


def plot_k_vs_accuracy(k_array, accuracy_array):
    plt.plot(k_array, accuracy_array, '-o')
    plt.xlabel('k value', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    plt.show()


# predict by K nearest weighted (by distance) neighbor
def k_weighted_nearest_neighbor(df, K):
    original_race_array = []
    predicted_race_array = []
    for i in range(0, df.shape[0]): # 0 ~ 1800

        # visualize the process
        clear_output(wait=True)
        print("Current Progress: ", np.round(i/df.shape[0] * 100, 2), "%")
        
        predicted_vector = [0] * 3
        for j in range(0, 2*K+1): # 0 ~ 100
            if (j == 0): 
                original_race = utils_final.index2race(utils_final.class2index(utils_final.fullName2Class(df.loc[i][j]))) #EX:70
                original_race_array.append(original_race)
            else:
                if (j % 2 == 1): # get classType
                    RaceUnitVector = utils_final.race2unitVector(utils_final.index2race(utils_final.class2index(utils_final.fullName2Class(df.loc[i][j]))))
                    print(RaceUnitVector)
                else: # get distance information
                    distance = df.loc[i][j]
                    if (distance == 0):
                        distance = 0.00000001

                    # add the lists element-wisely 
                    predicted_vector = list( map(add, predicted_vector, [k * (1/distance) for k in RaceUnitVector] ) ) #EX:predicted_vector=[1000,0,0]
                    print(predicted_vector)

        # get the max idx in the predicted_result
        # print(predicted_vector)
        predicted_race = (predicted_vector.index(max(predicted_vector))) #EX:predicted_race=0 
        if predicted_race == 0: # 0> 70
            predicted_race = 70
        elif predicted_race == 1:
            predicted_race = 80
        else: predicted_race = 90

        predicted_race_array.append(predicted_race)

    return original_race_array, predicted_race_array


# predict unknown
# ex: unknown = [EUL602, A12, 0.1, B23, 0.5, C45, 0.7, ...]
# output: [yellow, white, black] = [0.3, 0.5, 0.7]
def predict_race(unknown, d_star):
    original_race = utils_final.index2race(utils_final.class2index(utils_final.fullName2Class(unknown[0]))); 
    predicted_race_list = [0, 0, 0];
    neighbor_class = "";
    for idx in range(1, len(unknown)):
        if (idx % 2 == 1): # handle name
            # neighbor_class = toclass(unknown([idx]));
            neighbor_race = utils_final.index2race(utils_final.class2index(utils_final.fullName2Class(unknown[idx])));
            # print(neighbor_class)
        else: # handle distance
            # handle extra case, ex: distance = 0
            if (float(unknown[idx]) < 0.000000001): 
                # print("distance==0")
                unknown[idx] = "0.000000001";
            if (neighbor_race == 70):
                a = [i * 1/float(unknown[idx]) for i in [1, 0, 0]];
                predicted_race_list = list(map(add, predicted_race_list, a)); # predicted_race_list = predicted_race_list + a
                # print("a",a)
            if (neighbor_race == 80):
                b = [i * 1/float(unknown[idx]) for i in [0, 1, 0]];
                predicted_race_list = list(map(add, predicted_race_list, b)); # predicted_race_list = predicted_race_list + b
                # print("b",b)
            if (neighbor_race == 90):
                c = [i * 1/float(unknown[idx]) for i in [0, 0, 1]];
                predicted_race_list = list(map(add, predicted_race_list, c)); # predicted_race_list = predicted_race_list + c
                # print("c",c)
            if (float(unknown[idx]) > d_star): # check if distance is small enough, if bigger than d_star, then just get out
                break

    # normalize the predicted_list, ex: [0.3, 0.5, 0.7] -> [0.3/(0.3+0.5+0.7), 0.5/(0.3+0.5+0.7), 0.7/(0.3+0.5+0.7)] 
    predicted_race_list = [float(i)/sum(predicted_race_list) for i in predicted_race_list]
    return predicted_race_list;



def predict_subpop(unknown, d_star):
    original_class = utils_final.class2index(utils_final.fullName2Class(unknown[0])); 
    predicted_class_list = [0] * 35
    for idx in range(1, len(unknown)):
        if (idx % 2 == 1): # handle name
            neighbor_class_unitvector = utils_final.class2unitVector(utils_final.fullName2Class(unknown[idx]));
        
        else: # handle distance
            if (float(unknown[idx]) < 0.000000001): # handle extra case, ex: distance = 0
                unknown[idx] = "0.000000001";

            neighbor_class_unitvector = [i * 1/float(unknown[idx]) for i in neighbor_class_unitvector]; # neighbor_class_unitvector = 1/d * [...]
            predicted_class_list = list(map(add, predicted_class_list, neighbor_class_unitvector)); # predicted_class_list = predicted_class_list + neighbor_class_unitvector
            
            if (float(unknown[idx]) > d_star): # check if distance is small enough, if bigger than d_star, then just get out
                break

    # normalize the predicted_list, ex: [0.3, 0.5, 0.7] -> [0.3/(0.3+0.5+0.7), 0.5/(0.3+0.5+0.7), 0.7/(0.3+0.5+0.7)] 
    predicted_class_list = [float(i)/sum(predicted_class_list) for i in predicted_class_list]
    return predicted_class_list;




# lst = [1, 3, 4, 2] # after normalize -> [0.1, 0.3, ..]
# print(normalize(lst))

# unknown = ["EUL", "70", "0.1", "80", "0.5", "90", "0.7"];
# result = predict(unknown); # result = [0.3, 0, 0.7]
# result = normalize(result)
# print(result);

#---------------------------
def subpop2unitVector(className):

    # check the status of the string
    if not className:
        print("the className is EMPTY in class2unitVector()")
        time.sleep(5)

    className = className.upper()

    lst = [0] * 16

    if(className == "THANA" or className == "HAN" or className == "HAN_KO"): 
        lst[0] = 1;
    elif(className == "A" or className == "AMI"): 
        lst[1] = 1;
    elif(className == "P" or className == "PAI"): 
        lst[2] = 1;
    elif(className == "BS" or className == "PUY"): 
        lst[3] = 1;
    elif(className == "C" or className == "TSO"): 
        lst[4] = 1;
    elif(className == "T" or className == "ATA"): 
        lst[5] = 1;
    elif(className == "B" or className == "BUN"): 
        lst[6] = 1;
    elif(className == "S" or className == "SAI"): 
        lst[7] = 1;
    elif(className == "L" or className == "RUK"): 
        lst[8] = 1;
    elif(className == "Y" or className =="TAO"): 
        lst[9] = 1;
    elif(className == "HAK"): 
        lst[10] = 1;
    elif(className == "PIN"): 
        lst[11] = 1;
    elif(className == "J" or className == "AP"): 
        lst[12] = 1;
    elif(className == "KC"): 
        lst[13] = 1;
    elif(className == "EU" or className == "JN"): 
        lst[14] = 1;
    elif(className == "AUT" or className == "GER" or className == "FRENCH" or className == "UK" or className == "JQ" or className == "AY" or className == "RCRS"): 
        lst[15] = 1;
    
    else:
        print("error name in class2unitVector(): ")
        print("className: ", className)
        lst[0] = 1;
        time.sleep(5)
    return lst



#predict Asian subpopulation
def sample2subpopulation(unknown):
    # original = toclass(unknown[0]);
    
    original = unknown[0];
    print("original", original)
    predicted_list = [0]*16;
    neighbor_class = "";
    for idx in range(1, len(unknown)):
        if (idx % 2 == 1): # handle name
            # neighbor_class = toclass(unknown([idx]));
            neighbor_subpop = utils_final.fullName2Class(unknown[idx])
            neighbor_subpop_vector = utils_final.subpop2unitVector(utils_final.fullName2Class(unknown[idx]));
            print("neighbor_subpop", neighbor_subpop)
            # print(neighbor_class)
        else: # handle distance
            # handle extra case, ex: distance = 0
            print("unknown[idx]:",unknown[idx])
            if (float(unknown[idx])< 0.0000001): 
                print("distance==0")
                unknown[idx] = "0.000000001";
            if (neighbor_subpop == "THANA" or neighbor_subpop == "HAN" or neighbor_subpop == "HAN_KO"):
                a = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, a)); # predicted_list = predicted_list + a
                print("a",a)
            if (neighbor_subpop == "A" or neighbor_subpop == "AMI"):
                b = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, b)); # predicted_list = predicted_list + a
                print("b",b)
            if (neighbor_subpop == "P" or neighbor_subpop == "PAI"):
                c = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, c)); # predicted_list = predicted_list + a
                print("c",c)
            if (neighbor_subpop == "BS" or neighbor_subpop == "PUY"):
                d = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, d)); # predicted_list = predicted_list + a
                print("d",d)
            if (neighbor_subpop == "C" or neighbor_subpop == "TSO"):
                e = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, e)); # predicted_list = predicted_list + a
                print("e",e)
            if (neighbor_subpop == "T" or neighbor_subpop == "ATA"):
                f = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, f)); # predicted_list = predicted_list + a
                print("f",f)
            if (neighbor_subpop == "B" or neighbor_subpop == "BUN"):
                g = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, g)); # predicted_list = predicted_list + a
                print("g",g)
            if (neighbor_subpop == "S" or neighbor_subpop == "SAI"):
                h = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, h)); # predicted_list = predicted_list + a
                print("h",h)
            if (neighbor_subpop == "L" or neighbor_subpop == "RUK"):
                i = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, i)); # predicted_list = predicted_list + a
                print("i",i)
            if (neighbor_subpop == "Y" or neighbor_subpop == "TAO"):
                j = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, j)); # predicted_list = predicted_list + a
                print("j",j)
            if (neighbor_subpop == "HAK"):
                k = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, k)); # predicted_list = predicted_list + a
                print("k",k)
            if (neighbor_subpop == "PIN"):
                l = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, l)); # predicted_list = predicted_list + a
                print("l",l)
            if (neighbor_subpop == "J" or neighbor_subpop == "AP"):
                m = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, m)); # predicted_list = predicted_list + a
                print("m",m)
            if (neighbor_subpop == "KC"):
                n = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, n)); # predicted_list = predicted_list + a
                print("n",n)
            if (neighbor_subpop == "EU" or neighbor_subpop == "JN"):
                o = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, o)); # predicted_list = predicted_list + a
                print("o",o)
            if (neighbor_subpop == "AUT" or neighbor_subpop == "GER" or neighbor_subpop == "FRENCH" or neighbor_subpop == "UK" or neighbor_subpop == "JQ" or neighbor_subpop == "AY" or neighbor_subpop == "RCRS"):
                p = [i * 1/float(unknown[idx]) for i in neighbor_subpop_vector];
                predicted_list = list(map(add, predicted_list, p)); # predicted_list = predicted_list + a
                print("p",p)
    return predicted_list;




def distance_threshold_clustering(df, distance_threshold):
    original_race_array = []
    predicted_race_array = []
    for i in range(0, df.shape[0]):
        if (i % 200 == 0):
            print("processing to", i)


        neighborRaceType = []
        for j in range(0, df.shape[1]):
            if (j == 0): # original
                original_race = utils_final.index2race(utils_final.class2index(utils_final.fullName2Class(df.loc[i][j])))
                original_race_array.append(original_race)
            elif (j % 2 == 1): # neighbor
                neighborRaceType.append(utils_final.index2race(utils_final.class2index(utils_final.fullName2Class(df.loc[i][j]))))

            elif (float(df.loc[i][j]) % 2 == 0 and float(df.loc[i][j]) > distance_threshold):
                print("df.loc[i][j]: ", df.loc[i][j])
                print("j: ", j)
                break; 
        
        # get the most frequent element
        predicted_race = utils_final.getMostFrequentNum(neighborRaceType)
        predicted_race_array.append(predicted_race)

    (Y2Y, Y2W, Y2B, W2Y, W2W, W2B, B2Y, B2W, B2B) = analyze_indivudual_race_accuracy(original_race_array, predicted_race_array)
    d_to_perfect = distance_of_YWB_to_perfect(Y2Y, Y2W, Y2B, W2Y, W2W, W2B, B2Y, B2W, B2B)
    return d_to_perfect



# original_class_array: [70, 70, 80, 90, ...]
# predicted_class_array: [70, 80, 90, 90, ...]
# return: (Y2Y, Y2W, Y2B, W2Y, W2W, W2B, B2Y, B2W, B2B)
def analyze_individual_race_accuracy(original_race_array, predicted_race_array):
    number_of_70 = 0;
    number_of_80 = 0;
    number_of_90 = 0;
    num_70_70 = 0; # original: 70, predicted 70
    num_70_80 = 0; # original: 70, predicted 80
    num_70_90 = 0; # original: 70, predicted 90
    num_80_70 = 0; 
    num_80_80 = 0; 
    num_80_90 = 0;
    num_90_70 = 0; 
    num_90_80 = 0; 
    num_90_90 = 0;


    for i in range(0, len(original_race_array)):
        # calculate the total number
        if (original_race_array[i] == 70):
            number_of_70 = number_of_70 + 1
        if (original_race_array[i] == 80):
            number_of_80 = number_of_80 + 1
        if (original_race_array[i] == 90):
            number_of_90 = number_of_90 + 1

        # compare the result
        if (original_race_array[i] == 70 and predicted_race_array[i] == 70):
            num_70_70 = num_70_70 + 1;
        if (original_race_array[i] == 70 and predicted_race_array[i] == 80):
            num_70_80 = num_70_80 + 1;
        if (original_race_array[i] == 70 and predicted_race_array[i] == 90):
            num_70_90 = num_70_90 + 1;

        if (original_race_array[i] == 80 and predicted_race_array[i] == 70):
            num_80_70 = num_80_70 + 1;
        if (original_race_array[i] == 80 and predicted_race_array[i] == 80):
            num_80_80 = num_80_80 + 1;
        if (original_race_array[i] == 80 and predicted_race_array[i] == 90):
            num_80_90 = num_80_90 + 1;

        if (original_race_array[i] == 90 and predicted_race_array[i] == 70):
            num_90_70 = num_90_70 + 1;
        if (original_race_array[i] == 90 and predicted_race_array[i] == 80):
            num_90_80 = num_90_80 + 1;
        if (original_race_array[i] == 90 and predicted_race_array[i] == 90):
            num_90_90 = num_90_90 + 1; 

    # print(number_of_70)
    # print(number_of_80)
    # print(number_of_90)
    # print("Y2Y_accuracy:", float(num_70_70)/number_of_70)
    # print("Y2W_accuracy:", float(num_70_80)/number_of_70)
    # print("Y2B_accuracy:", float(num_70_90)/number_of_70)

    # print("W2Y_accuracy:", float(num_80_70)/number_of_80)
    # print("W2W_accuracy:", float(num_80_80)/number_of_80)
    # print("W2B_accuracy:", float(num_80_90)/number_of_80)

    # print("B2Y_accuracy:", float(num_90_70)/number_of_90)
    # print("B2W_accuracy:", float(num_90_80)/number_of_90)
    # print("B2B_accuracy:", float(num_90_90)/number_of_90)

    #define return value
    if (number_of_70 == 0):
        num_70_70=0
        num_70_80=0
        num_70_90=0
    else:
        Y2Y_accuracy = float(num_70_70)/number_of_70
        Y2W_accuracy = float(num_70_80)/number_of_70
        Y2B_accuracy = float(num_70_90)/number_of_70

    if (number_of_80 == 0):
        num_80_70=0
        num_80_80=0
        num_80_90=0
    else:
        W2Y_accuracy = float(num_80_70)/number_of_80
        W2W_accuracy = float(num_80_80)/number_of_80
        W2B_accuracy = float(num_80_90)/number_of_80

    if (number_of_90 == 0):
        num_90_70=0
        num_90_80=0
        num_90_90=0
    else:
        B2Y_accuracy = float(num_90_70)/number_of_90
        B2W_accuracy = float(num_90_80)/number_of_90
        B2B_accuracy = float(num_90_90)/number_of_90


    return Y2Y_accuracy, Y2W_accuracy, Y2B_accuracy, W2Y_accuracy, W2W_accuracy, W2B_accuracy, B2Y_accuracy, B2W_accuracy, B2B_accuracy



def distance_of_YWB_to_perfect(Y2Y_accuracy, Y2W_accuracy, Y2B_accuracy, W2Y_accuracy, W2W_accuracy, W2B_accuracy, B2Y_accuracy, B2W_accuracy, B2B_accuracy):
    distance = (1-Y2Y_accuracy)**2 + (1-W2W_accuracy)**2 + (1-B2B_accuracy)**2 + Y2W_accuracy**2 + Y2B_accuracy**2 + W2Y_accuracy**2 + W2B_accuracy**2 + B2Y_accuracy**2 + B2W_accuracy**2
    return distance


# YWB matrix = 
#[y2y(1), y2w(0), y2b(0)]
#[w2y(0), w2w(1), w2b(0)]
#[b2y(0), b2w(0), b2b(1)]   <- perfect matrix


# randomly create training/testing data set
def create_training_testing_data(df, training_percent, testing_percent):
    df_training = df.sample(frac = training_percent, replace = True)
    df_testing = df.sample(frac = testing_percent, replace = True)
    return df_training, df_testing


