import pandas as pd
import numpy as np
from scipy.stats import norm


class NBClassifier(object):
    def __init__(self,input_file):
        self.dataset = pd.read_csv(input_file)

    def get_prior_prob(self,classifier):
         num_class_rep = self.dataset["sex"].value_counts()[classifier]
         total_count = len(self.dataset)
         return num_class_rep / total_count

    def get_posterior_prob(self,X,classifier):
        sex = self.dataset.loc[self.dataset['sex'] == classifier]
        mn = sex.mean().values
        sd = sex.std().values
        prob =  norm.pdf(X, loc=mn, scale=sd)
        return prob.prod()

    def get_conditional_prob(self,X,classifier):
        posterior_prob = {}
        posterior_prob["male"] = self.get_posterior_prob(X,"male")
        posterior_prob["female"] = self.get_posterior_prob(X,"female")

        total_prob = ( posterior_prob["male"] * self.get_prior_prob("male") +
                       posterior_prob["female"] * self.get_prior_prob("female"))

        return (posterior_prob[classifier] * self.get_prior_prob(classifier))  / total_prob


def main():
    X = np.array([5.4,140,10])
    classifier = NBClassifier("input.csv")
    male_prob = classifier.get_conditional_prob(X,"male")
    female_prob = classifier.get_conditional_prob(X,"female")

    print(f"male: {male_prob.round(3)*100}%")
    print(f"female: {female_prob.round(3)*100}%\n")

    prediction = "male" if male_prob > female_prob else "female"
    print(f"Sample classified as {prediction}")


if __name__ == "__main__":
    main()
