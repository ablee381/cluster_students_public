import requests
import os
import time


def load_latex(standard, pdf_id, outDir='tex_zips'):
    outFile = standard + '.tex.zip'
    if outFile not in os.listdir(outDir):
        headers = {
            "app_id": os.getenv('MATHPIX_ID'),
            "app_key": os.getenv('MATHPIX_KEY')
        }
        # get LaTeX zip file
        url = "https://api.mathpix.com/v3/pdf/" + pdf_id + ".tex"
        response = requests.get(url, headers=headers)
        print('write the tex file')
        with open(outDir + '/' + outFile, "wb") as f:
            f.write(response.content)
    else:
        print('Already written')


def check_status(pdf_id):
    url = 'https://api.mathpix.com/v3/converter/' + pdf_id
    json_dict = requests.get(url,
                             headers={
                                 "app_id": os.getenv('MATHPIX_ID'),
                                 "app_key": os.getenv('MATHPIX_KEY')
                             }
                             ).json()
    return json_dict['status']


if __name__ == '__main__':
    print(check_status('2023_08_28_566d224c9a5c598df557g'))
    # f = open('demo/pdf_id.txt', 'r')
    # line = f.readline().strip()
    # lineParts = line.split('\t')
    # print(lineParts)
    # assert(len(lineParts) == 2)
    # test_name = lineParts[0]
    # pdf_id = lineParts[1].split(':')[1][1:-2]
    # load_latex(test_name, pdf_id, outDir='demo')

    # # pdf_id.txt has 55 lines. I counted and that is what I was expecting.
    # # I don't want infinite looping api calls, so no while loops
    # for i in range(55):
    #     line = f.readline().strip()
    #     lineParts = line.split('\t')
    #     print(lineParts)
    #     assert(len(lineParts) == 2)
    #     standard = lineParts[0]
    #     pdf_id = lineParts[1].split(':')[1][1:-2]
    #     load_latex(standard, pdf_id)
    #     time.sleep(1.5)

# # Check status of conversion
# json_dict = requests.get('https://api.mathpix.com/v3/converter/2023_07_03_01209c7a1a0964e85bb5g',
#                          headers={
#                              "app_id": os.getenv('MATHPIX_ID'),
#                              "app_key": os.getenv('MATHPIX_KEY')
#                          }
#                          ).json()
# print('status', json_dict)
