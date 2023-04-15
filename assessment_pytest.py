import pytest
import numpy as np

from assessmentClass import Results

@pytest.fixture()
def my_results():
    return Results('test_illuminate.xlsx')

@pytest.fixture()
def potts_results():
    return Results('test_potts.xlsx')

def test_update1(my_results):
    my_results.qNum = 17
    my_results.updateCutoff()
    assert my_results.cutoff == my_results.qNum*my_results.guessRate+\
           np.sqrt(my_results.qNum*my_results.guessRate*(1-my_results.guessRate))


def test_read_illuminate(my_results):
    my_results.read_illuminate()
    assert sum(my_results.data[1]) == 1
    assert sum(my_results.data[5]) == 3
    my_results.norm_illuminate()
    assert sum(my_results.data[1]) == 0

def test_read_potts(potts_results):
    potts_results.read_potts()
    assert sum(potts_results.data['1']) == 5
    assert sum(potts_results.data['8']) == 2