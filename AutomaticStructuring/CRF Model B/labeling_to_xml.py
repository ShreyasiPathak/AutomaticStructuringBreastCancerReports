from xml.etree.ElementTree import Element, SubElement, Comment, tostring, ElementTree
from xml.dom import minidom
import sys

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def test_onPredicted(beg,end,phrase,preLabel,currentNode,currentNode1):
    #print currentNode
    phrase2=phrase[beg:end]
    j=0
    flag=0
    flag1=0
    for i in range(beg,end):
        preLabelSplit=preLabel[i].split('/')
        pointer=currentNode.split('/')
        if len(preLabelSplit)>len(pointer)-1:
            preLabelOne=preLabelSplit[len(pointer)-1]
        else:
            preLabelOne='O'
        #data1=preTokenList[j][1]
        print preLabelOne
        labelStartPres=preLabelOne.split('-')[0]
        print labelStartPres
        #labelChildStartPres=preLabelTwo.split('-')[0]
        labelEndPres=preLabelOne.split('-')[len(preLabelOne.split('-'))-1]
        #print preLabels
        '''if labelEndPres!='O' or preLabels[i][0]=='':
            preLabels[i][0]=preLabels[i][0]+"/"+labelEndPres
            preLabels[i][1]=preLabels[i][1]+"/"+pre1'''
        if j!=len(phrase2)-1:
            preLabelSplitNext=preLabel[i+1].split('/')
            if len(preLabelSplitNext)>len(pointer)-1:
                preLabelOneNext=preLabelSplitNext[len(pointer)-1]
            else:
                preLabelOneNext='O'
            labelStartNext=preLabelOneNext.split('-')[0]
        else:
            labelStartNext=None
        if labelStartPres=='B':
            beg1=i
            if labelStartNext=='B' or labelStartNext=='O' or j==len(phrase2)-1:
                end1=i
                flag=1
                child=str(currentNode)+"/"+labelEndPres
                child1 = SubElement(currentNode1, labelEndPres)
                #print child1
                #print phrase[beg:end+1]
                if labelEndPres=="positive_finding" or labelEndPres=="negative_finding" \
                or labelEndPres=="mass" or labelEndPres=="calcification" \
                or labelEndPres=="asymmetry" or labelEndPres=="architectural_distortion" \
                or child=="report/positive_finding/associated_features" or child=="report/negative_finding/associated_features":
                    test_onPredicted(beg1,end1+1,phrase,preLabel,child,child1)
                    #print "pf:",preLabel
                else:
                    #print " ".join(phrase[beg:end+1])
                    child1.text=" ".join(phrase[beg1:end1+1])
        elif labelStartPres=='I':
            if labelStartNext=='B' or labelStartNext=='O' or j==len(phrase2)-1:
                end1=i
                flag=1
                child=str(currentNode)+"/"+labelEndPres
                child1 = SubElement(currentNode1, labelEndPres)
                if labelEndPres=="positive_finding" or labelEndPres=="negative_finding" \
                or labelEndPres=="mass" or labelEndPres=="calcification" \
                or labelEndPres=="asymmetry" or labelEndPres=="architectural_distortion" \
                or child=="report/positive_finding/associated_features" or child=="report/negative_finding/associated_features":
                    test_onPredicted(beg1,end1+1,phrase,preLabel,child,child1)
                    #print "pf:",preLabel
                else:
                    child1.text=" ".join(phrase[beg1:end1+1])
        if labelStartPres=='O':
            if flag1==0:
                beg1=i
                flag1=1
            #print phrase[i]
            if labelStartNext=='B' or j==len(phrase2)-1:
                end1=i
                flag1=0
                #print phrase[i]
                if currentNode=="report":
                    print currentNode1
                    print phrase[beg1:end1+1]
                    child1=SubElement(currentNode1, labelEndPres)
                    child1.text=" ".join(phrase[beg1:end1+1])
                elif flag==1:
                    #print phrase[beg:end+1]
                    #print "in:",child1
                    child1.tail=" ".join(phrase[beg1:end1+1])
                elif flag==0:
                    currentNode1.text=" ".join(phrase[beg1:end1+1])
                flag=0
        j=j+1

def mainFunc(tokenList,cascPre):
    top = Element('radiology_reports')
    for i in range(len(tokenList)):
        node=SubElement(top,"report")
        test_onPredicted(0,len(tokenList[i]),tokenList[i],cascPre[i],"report",node)
    f=open("predicted_xml",'w')
    #ElementTree(top).write(f)
    #print prettify(top)
    f.write(prettify(top))

'''top = Element('top')

comment = Comment('Generated for PyMOTW')
top.append(comment)

child = SubElement(top, 'child')
child.text = 'This child contains text.'

child_with_tail = SubElement(top, 'child')
child_with_tail.text = 'This child has regular text.'
child_with_tail.tail = 'And "tail" text.'

child_with_entity_ref = SubElement(top, 'child_with_entity_ref')
child_with_entity_ref.text = 'This & that'

print prettify(top)'''
