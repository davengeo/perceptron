(ns perceptron.core-test
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.test :refer :all]
            [perceptron.core :as perceptron]))

(deftest prepare-batch
  (testing "prepare batch produces a matrix of n samples"
    (is (matrix? (perceptron/prepare-batch '(1 2 3) 3 1)))
    (is (= 1 (ncols (perceptron/prepare-batch '(1 2 3) 3 1))))
    (is (= 3 (mrows (perceptron/prepare-batch '(1 2 3) 3 1))))
    ))



(deftest map-by-layer1
  (testing "map over a vector"
    (is ())
    ))