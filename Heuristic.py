import numpy
import math
import random
import copy

numnodesgen, numnodesexp = 0, 0


class Node:
    def __init__(self, input_data, previous_data, value_G, heuValue, function_value) -> None:
        self.input_data,self.valueold,self.Value_G,self.Value_h,self.total_functionval = input_data,previous_data,value_G,heuValue,function_value

    def suc_gen_function(self):
        i, j = self.value_position_find('0')
        output = []
        if (i + 1 < len(self.input_data)):output.append(self.generateChildNode(self.input_data, i, j, i + 1, j))
        if (j + 1 < len(self.input_data)):output.append(self.generateChildNode(self.input_data, i, j, i, j + 1))
        if (i - 1 >= 0):output.append(self.generateChildNode(self.input_data, i, j, i - 1, j))
        if (j - 1 >= 0):output.append(self.generateChildNode(self.input_data, i, j, i, j - 1))
        return output

    def value_position_find(self, input_value_space):
        check = self.input_data
        for i in range(len(self.input_data)):
            for j in range(len(self.input_data)):
                if (check[i][j] == input_value_space):
                    return i, j

    def generateChildNode(self, input, oldcoordinateone, oldcoordinatetwo, changedcoordinateone, changedcoordinatetwo):
        copy_data = copy.deepcopy(input)
        temperoryval = copy_data[oldcoordinateone][oldcoordinatetwo]
        copy_data[oldcoordinateone][oldcoordinatetwo] = input[changedcoordinateone][changedcoordinatetwo]
        copy_data[changedcoordinateone][changedcoordinatetwo] = temperoryval
        return Node(copy_data, None, self.Value_G + 1, 0, 0)


class Puzzle:
    def __init__(self, puzzle_size) -> None:
        self.puzzle_size,self.Puzzle_front,self.puzzle_expand = puzzle_size,[],[]
        
    def Input_matrix(self):
        arrayinput = []
        for i in range(self.puzzle_size):
            temporery_store = input().split(" ")
            arrayinput.append(temporery_store)
        return arrayinput
    
    def generate_random(self):
        goal1 = [1,2,3,4,5,6,7,8,0]
        rand=""
        list = []
        for i in range(0,9):
            num1 = random.randint(0,len(goal1)-1)
            rand=rand+str(goal1[num1])
            goal1.remove(goal1[num1])
        list[:0] = rand
        count=0
        for i in range(0,len(list)):
            if list[i]!=0:
                for j in range(i+1,len(list)):
                    if list[i]>list[j]:
                        count=count+1                  
        if count%2==0:
            return rand
        else:
            return self.generate_random()


    def strtomatrix(self,str):  
        length = len(str);   
        n = 3;  
        temp = 0;  
        chars = int(length/n);  
        equalStr = [];   
        if(length % n != 0):  
            print("string can't be divided into " + str(n) +" equal parts.");  
        else:  
            for i in range(0, length, chars):  
                part = str[ i : i+chars];  
                equalStr.append(part);  
        res = [list(sub) for sub in equalStr]
        return res
        
    def print_array_matrix(self, input):
        for i in range(len(input)):
            print(input[i])

    def output_current_index(self, input_current):
        for index, node in enumerate(self.Puzzle_front):
            if (numpy.array_equal(input_current.input_data, node.input_data)):
                return index
        return None

    def Sol_checker(self, initial, input):
        startInputArray,FinaloutputArray = numpy.array(initial).flatten(),numpy.array(input).flatten()
        initial_state_parity = self.parity_check_value(startInputArray)
        goal_state_parity = self.parity_check_value(FinaloutputArray)
        if initial_state_parity == goal_state_parity:
            return True
        else:
            return False

    def parity_check_value(self, input_state):
        number_of_inversions,input_state = 0,input_state[input_state != "0"]
        for i in range(9):
            for j in range(i + 1, 8):
                if input_state[i] > input_state[j]:
                    number_of_inversions = number_of_inversions + 1
        if number_of_inversions % 2 == 0:
            return "even"
        else:
            return "odd"

    def matrix_solve2(self):
        print("Enter values:")
        start_Array_matrix = self.Input_matrix()
        final_gole_matrix = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '0']]
        is_ckeck_solveable = self.Sol_checker(start_Array_matrix, final_gole_matrix)
        if not is_ckeck_solveable:
            return None
        print("Enter your value of heuristic: 1.Number of tiles out of place 2.EuclideanDistance 3.CityBlockDistance Distance 4.Diagonal Distance :")
        print("Enter your value of inadmmisable heuristic: 5.Number of tiles out of place 6.EuclideanDistance 7.CityBlockDistance Distance 8.Diagonal Distance :")
        heuristic_val_checker = input()
        if heuristic_val_checker == "1":
            output_heu_fun = Number_of_tiles_out_of_place(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "2":
            output_heu_fun = EuclideanDistance(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "3":
            output_heu_fun = HeuCityBlockDistance(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "4":
            output_heu_fun = Diagonal_Distance(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "5":
            output_heu_fun = inadmissiablenooftiles(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "6":
            output_heu_fun = inadmissableEuclideanDistance(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "7":
            output_heu_fun = inadmissableHeuCityBlockDistance(start_Array_matrix, final_gole_matrix)
        else:
            output_heu_fun = inadmmisableDiagonal_Distance(start_Array_matrix, final_gole_matrix)    
        start_Array_matrix = Node(start_Array_matrix, None, 0, 0, 0)
        start_Array_matrix.total_functionval = output_heu_fun.HeuvalCal(start_Array_matrix.input_data) + start_Array_matrix.Value_G
        self.Puzzle_front.append(start_Array_matrix)
        numnodesgen = 1
        while (len(self.Puzzle_front) > 0):
            output = self.Puzzle_front.pop(0)
            self.puzzle_expand.append(output.input_data)
            if (numpy.array_equal(output.input_data, final_gole_matrix)):
                print("At the time reached goal state")
                print("Number of nodes Generated , Number of nodes Expanded=", len(self.Puzzle_front) + len(self.puzzle_expand),",",len(self.puzzle_expand))
                print("Number of steps required for optimal solution = ",output.Value_G)
                return output

            for CHILD in output.suc_gen_function():
                if (not (any(numpy.array_equal(CHILD.input_data, x) for x in self.puzzle_expand))):
                    CHILD.total_functionval = output_heu_fun.HeuvalCal(CHILD.input_data) + CHILD.Value_G
                    CHILD.valueold,current_index_of_child = output,self.output_current_index(CHILD)
                    if (current_index_of_child != None):
                        if (output.total_functionval < self.Puzzle_front[current_index_of_child].total_functionval):
                            self.Puzzle_front[current_index_of_child] = CHILD
                    else:
                        self.Puzzle_front.append(CHILD)
            self.Puzzle_front.sort(key=lambda data: data.total_functionval, reverse=False)
        return None
    
    def matrix_solve(self):
        inputramdongen=self.generate_random()
        start_Array_matrix,final_gole_matrix = self.strtomatrix(inputramdongen),[['1', '2', '3'], ['4', '5', '6'], ['7', '8', '0']]
        is_ckeck_solveable = self.Sol_checker(start_Array_matrix, final_gole_matrix)
        if not is_ckeck_solveable:
            return self.matrix_solve()
        print("Enter your value of heuristic: 1.Number of tiles out of place 2.EuclideanDistance 3.CityBlockDistance Distance 4.Diagonal Distance :")
        print("Enter your value of inadmmisable heuristic: 5.Number of tiles out of place 6.EuclideanDistance 7.CityBlockDistance Distance 8.Diagonal Distance :")
        heuristic_val_checker = input()
        if heuristic_val_checker == "1":
            output_heu_fun = Number_of_tiles_out_of_place(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "2":
            output_heu_fun = EuclideanDistance(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "3":
            output_heu_fun = HeuCityBlockDistance(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "4":
            output_heu_fun = Diagonal_Distance(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "5":
            output_heu_fun = inadmissiablenooftiles(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "6":
            output_heu_fun = inadmissableEuclideanDistance(start_Array_matrix, final_gole_matrix)
        elif heuristic_val_checker == "7":
            output_heu_fun = inadmissableHeuCityBlockDistance(start_Array_matrix, final_gole_matrix)
        else:
            output_heu_fun = inadmmisableDiagonal_Distance(start_Array_matrix, final_gole_matrix)    
        start_Array_matrix = Node(start_Array_matrix, None, 0, 0, 0)
        start_Array_matrix.total_functionval = output_heu_fun.HeuvalCal(start_Array_matrix.input_data) + start_Array_matrix.Value_G
        self.Puzzle_front.append(start_Array_matrix)
        numnodesgen = 1
        while (len(self.Puzzle_front) > 0):
            output = self.Puzzle_front.pop(0)
            self.puzzle_expand.append(output.input_data)
            if (numpy.array_equal(output.input_data, final_gole_matrix)):
                numnodesgen= len(self.Puzzle_front) + len(self.puzzle_expand)
                numnodesexp=len(self.puzzle_expand)
                print("At the time reached goal state")
                print("Number of nodes Generated , Number of nodes Expanded=",numnodesgen,",",numnodesexp)
                print("Number of steps required for optimal solution = ",output.Value_G)
                return output

            for CHILD in output.suc_gen_function():
                if (not (any(numpy.array_equal(CHILD.input_data, x) for x in self.puzzle_expand))):
                    CHILD.total_functionval = output_heu_fun.HeuvalCal(CHILD.input_data) + CHILD.Value_G
                    CHILD.valueold,current_index_of_child = output,self.output_current_index(CHILD)
                    if (current_index_of_child != None):
                        if (output.total_functionval < self.Puzzle_front[current_index_of_child].total_functionval):
                            self.Puzzle_front[current_index_of_child] = CHILD
                    else:
                        self.Puzzle_front.append(CHILD)
            self.Puzzle_front.sort(key=lambda data: data.total_functionval, reverse=False)
        return None


class Number_of_tiles_out_of_place:
    def __init__(self, start, final_goal) -> None:
        self.start,self.final_goal = start,final_goal

    def HeuvalCal(self, present_state):
        Number_of_tiles_out_of_place = 0
        for i in range(3):
            for j in range(3):
                if present_state[i][j] != self.final_goal[i][j] and (present_state[i][j] != "0"):
                    Number_of_tiles_out_of_place =Number_of_tiles_out_of_place+ 1
        return Number_of_tiles_out_of_place


class HeuCityBlockDistance:
    def __init__(self, start, final_goal) -> None:
        self.start,self.final_goal = start, final_goal

    def finalTileCoord(self, input_tile):
        for i in range(3):
            for j in range(3):
                if (self.final_goal[i][j] == input_tile):
                    return i, j

    def HeuvalCal(self, present_state):
        output = 0
        for i in range(3):
            for j in range(3):
                present_tile = present_state[i][j]
                if present_state[i][j] != "0":
                    valx1,valy1 = i,j
                    valx2, valy2 = self.finalTileCoord(present_tile)
                    val = abs(valx1 - valx2) + abs(valy1 - valy2)
                    output = output + val
        return output

class EuclideanDistance:
    def __init__(self, start, final_goal) -> None:
        self.start, self.final_goal = start, final_goal

    def finalTileCoord(self, input_tile):
        for i in range(3):
            for j in range(3):
                if (self.final_goal[i][j] == input_tile):
                    return i, j

    def HeuvalCal(self, present_state):
        output = 0
        for i in range(3):
            for j in range(3):
                present_tile = present_state[i][j]
                if present_state[i][j] != "0":
                    valx1, valy1 = i, j
                    valx2, valy2 = self.finalTileCoord(present_tile)
                    val = math.sqrt((valx1 - valx2)**2 + (valy1 - valy2)**2)
                    output = output + val
        return output
class Diagonal_Distance:
    def __init__(self, start, final_goal) -> None:
        self.start, self.final_goal = start, final_goal

    def finalTileCoord(self, input_tile):
        for i in range(3):
            for j in range(3):
                if (self.final_goal[i][j] == input_tile):
                    return i, j

    def HeuvalCal(self, present_state):
        output = 0
        for i in range(3):
            for j in range(3):
                present_tile = present_state[i][j]
                if present_state[i][j] != "0":
                    valx1, valy1 = i, j
                    valx2, valy2 = self.finalTileCoord(present_tile)
                    D,D2=1,1.414
                    val =abs(valx1 - valx2) + abs(valy1 - valy2)+ (D2 - 2 * D)*min((abs(valx1 - valx2)),(abs(valy1 - valy2)))
                    output = output + val
        return output
class inadmissiablenooftiles:
    def __init__(self, start, final_goal) -> None:
        self.start,self.final_goal = start,final_goal

    def HeuvalCal(self, present_state):
        Number_of_tiles_out_of_place = 0
        for i in range(3):
            for j in range(3):
                if present_state[i][j] != self.final_goal[i][j] and (present_state[i][j] != "0"):
                    Number_of_tiles_out_of_place =Number_of_tiles_out_of_place+ 1
        return 3*Number_of_tiles_out_of_place


class inadmissableHeuCityBlockDistance:
    def __init__(self, start, final_goal) -> None:
        self.start,self.final_goal = start, final_goal

    def finalTileCoord(self, input_tile):
        for i in range(3):
            for j in range(3):
                if (self.final_goal[i][j] == input_tile):
                    return i, j

    def HeuvalCal(self, present_state):
        output = 0
        for i in range(3):
            for j in range(3):
                present_tile = present_state[i][j]
                if present_state[i][j] != "0":
                    valx1,valy1 = i,j
                    valx2, valy2 = self.finalTileCoord(present_tile)
                    val = abs(valx1 - valx2) + abs(valy1 - valy2)
                    output = output + val
        return 3*output

class inadmissableEuclideanDistance:
    def __init__(self, start, final_goal) -> None:
        self.start, self.final_goal = start, final_goal

    def finalTileCoord(self, input_tile):
        for i in range(3):
            for j in range(3):
                if (self.final_goal[i][j] == input_tile):
                    return i, j

    def HeuvalCal(self, present_state):
        output = 0
        for i in range(3):
            for j in range(3):
                present_tile = present_state[i][j]
                if present_state[i][j] != "0":
                    valx1, valy1 = i, j
                    valx2, valy2 = self.finalTileCoord(present_tile)
                    val = math.sqrt((valx1 - valx2)**2 + (valy1 - valy2)**2)
                    output = output + val
        return 3*output
class inadmmisableDiagonal_Distance:
    def __init__(self, start, final_goal) -> None:
        self.start, self.final_goal = start, final_goal

    def finalTileCoord(self, input_tile):
        for i in range(3):
            for j in range(3):
                if (self.final_goal[i][j] == input_tile):
                    return i, j

    def HeuvalCal(self, present_state):
        output = 0
        for i in range(3):
            for j in range(3):
                present_tile = present_state[i][j]
                if present_state[i][j] != "0":
                    valx1, valy1 = i, j
                    valx2, valy2 = self.finalTileCoord(present_tile)
                    D,D2=1,1.414
                    val =abs(valx1 - valx2) + abs(valy1 - valy2)+ (D2 - 2 * D)*min((abs(valx1 - valx2)),(abs(valy1 - valy2)))
                    output = output + val
        return 3*output
    
    
    
print("Enter your value of 1 heuristic, 2.n heuristic output==ratio :")
heuristic_val = input()
if heuristic_val == "1":
    input_matrix = Puzzle(3)
    output = input_matrix.matrix_solve()
    if output == None:
        print("none")
    else:
        print("Best Path to Goal State:\n")
        outputpath = []
        while (output != None):
            outputpath.append(output.input_data)
            output = output.valueold
        outputpath.reverse()
        while (len(outputpath) > 1):
            path = outputpath.pop(0)
            input_matrix.print_array_matrix(path)
            print(" ")
        path = outputpath.pop(0)
        input_matrix.print_array_matrix(path)
    print(" ")
elif heuristic_val == "2":
    lis=[]
    print("How many n random generated cases")
    n=int(input())
    for i in range(n):
        input_matrix = Puzzle(3)
        output = input_matrix.matrix_solve()
        cost_opt=output.Value_G
        if output == None:
            print("none")
        else:
            print("Best Path to Goal State:\n")
            outputpath = []
            while (output != None):
                outputpath.append(output.input_data)
                output = output.valueold
            outputpath.reverse()
            while (len(outputpath) > 1):
                path = outputpath.pop(0)
                input_matrix.print_array_matrix(path)
                print(" ")
            path = outputpath.pop(0)
            input_matrix.print_array_matrix(path)
        print(" ")

        input_matrix = Puzzle(3)
        output = input_matrix.matrix_solve2()
        cost_non_opt=output.Value_G
        if output == None:
            print("none")
        else:
            print("Best Path to Goal State:\n")
            outputpath = []
            while (output != None):
                outputpath.append(output.input_data)
                output = output.valueold
            outputpath.reverse()
            while (len(outputpath) > 1):
                path = outputpath.pop(0)
                input_matrix.print_array_matrix(path)
                print(" ")
            path = outputpath.pop(0)
            input_matrix.print_array_matrix(path)
        print(" ")
    ratio=cost_opt/cost_non_opt
    print("Ratio is {}".format(ratio))
    lis.append(ratio)
    sum1=sum(lis)
    print("Average of the ratios is {}".format(sum1/n))