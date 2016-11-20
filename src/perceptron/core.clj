(ns perceptron.core
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.data.generators :as gen]))

(defn rnd-flts [n] (take n (repeatedly #(- (* (gen/float) 2) 1))))
(defn fill-with [n] (take n (cycle '(1 0))))
(defn init-layer [n m] (sge n m (fill-with (* n m))))

; refactor to multipethod or use a protocol
; (defmulti sigmoid :mode)
(def sigmoid (fn ^double [^double x] (/ 1 (+ 1 (Math/exp (- x))))))
(def training (fn ^double [^double x] (* x (- 1 x))))

(defn mm-map! [mm fnx]
  (for [x (range (mrows mm))]
    (for [y (range (ncols mm))]
      (alter! mm x y fnx))))

; create a record with all the dimensions and biases
(def input 4)
(def output 4)
(def neurons0 4)
(def neurons1 30)
(def neurons2 10)

(def syn0 (init-layer input neurons0))
(def syn1 (init-layer neurons0 neurons1))
(def syn2 (init-layer neurons1 neurons2))
(def syn3 (init-layer neurons2 output))

(defn l1 [x] (mm-map! (mm syn0 x) sigmoid))
(defn l2 [x] (mm-map! (mm syn1 (l1 x)) sigmoid))
(defn l3 [x] (mm-map! (mm syn2 (l2 x)) sigmoid))
(defn l4 [x] (mm-map! (mm syn3 (l3 x)) sigmoid))

(def my-vector (sge 4 4 (rnd-flts 16)))

(println (l2 my-vector))
