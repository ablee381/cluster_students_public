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

