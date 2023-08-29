import requests
import json
import os
import time


def post_mathpix(fileName, sourceDir='training_questions', outFile='pdf_id.txt', checkDir='tex_zips'):
    """uploads a test pdf to mathpix for conversion to LaTex"""
    # Check if we've already done the conversion
    test_name = fileName[:-4]  # get rid of the .pdf
    print(test_name)
    if test_name + '.tex.zip' not in os.listdir(checkDir):
        options = dict(conversion_formats={"tex.zip": True},
                       math_inline_delimiters=["$", "$"],
                       rm_spaces=True)
        r = requests.post("https://api.mathpix.com/v3/pdf",
                          headers={
                              "app_id": os.getenv('MATHPIX_ID'),
                              "app_key": os.getenv('MATHPIX_KEY')
                          },
                          data={
                              "options_json": json.dumps(options)
                          },
                          files={
                              "file": open(sourceDir + '/' + fileName, "rb")
                          }
                          )
        f = open(checkDir + '/' + outFile, 'a')
        f.write(test_name + '\t' + r.text + '\n')
        f.close()
        print(r.text.encode("utf8"))
        return r.text.split(':')[1][1:-2]
    else:
        print('Already processed this pdf')


if __name__ == '__main__':
    file_name = 'unit_test.pdf'
    post_mathpix(file_name, sourceDir='demo', checkDir='demo')
    # fileList = os.listdir('training_questions')
    # for i in range(4, len(fileList)):
    #     fileName = fileList[i]
    #     post_mathpix(fileName)
    #     time.sleep(1.5)
