def relative(r, name1, name2):
    if not r or not r[0] or len(r[0]) != 3:
        return []
    dicList = {}
    for relation in r:
        if relation[0] not in dicList:
            dicList[relation[0]] = {}
        if relation[2] not in dicList[relation[0]]:
            dicList[relation[0]][relation[2]] = []
        dicList[relation[0]][relation[2]].append(relation[1])
    visited = set()
    output = []
    currentRelation = [name1]
    def helper(name):
        if name in visited: return
        if name == name2:
            output.append(" ".join(currentRelation))
            return
        visited.add(name)
        # currentRelation.append(name)
        for secondName in dicList[name]:
            for nameRelationSecondName in dicList[name][secondName]:
                currentRelation.append(nameRelationSecondName)
                currentRelation.append(secondName)
                helper(secondName)
                currentRelation.pop()
                currentRelation.pop()
    helper(name1)
    return output

test = [
["B", "brother", "L"],
["B", "son", "H"],
["M", "wife", "H"],
["L", "daugtor", "H"]
]
print relative(test, "B", "H")
