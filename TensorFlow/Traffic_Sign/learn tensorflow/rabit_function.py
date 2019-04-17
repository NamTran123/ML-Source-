def count_occurences( lit , a):
    return len([x for x  in  lit if x == a and  type(x)== type(a)])

print(count_occurences([1,2,3,1,1,1,3] ,1))


def check_upcase(string1):
    return string1  == string1.upcase()