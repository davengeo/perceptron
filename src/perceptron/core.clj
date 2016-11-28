(ns perceptron.core
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.data.generators :as gen]))

(defn rnd-flts [n] (take n (repeatedly #(- (* (gen/float) 2) 1))))
(defn init-layer [n m] (sge n m (rnd-flts (* n m))))

(def sigmoid (fn ^double [^double x] (/ 1 (+ 1 (Math/exp (- x))))))
(def training (fn ^double [^double x] (* x (- 1 x))))

(defn mm-map! [mm fnx]
  (do (for [x (range (mrows mm))]
    (for [y (range (ncols mm))]
      (alter! mm x y fnx)))
      (identity mm)))

(defn h* [xx yy]
  (do
    (if (not (and (= (mrows xx) (mrows yy)) (= (ncols xx) (ncols yy))))
      (throw (Exception. "not compatible args")))
    (let [result (zero xx)]
    (for [x (range (mrows xx))]
        (for [y (range ncols xx)]
          (entry! result x y (* (entry xx x y) (entry yy x y)))
          ))
    (identity result))))


(def my-input (sge 4 4 (rnd-flts 16)))
;4x4
(def my-output (sge 4 4 (rnd-flts 16)))
;4x4

; create a record with all the dimensions
(def input 4)
(def output 4)
(def neurons0 4)
(def neurons1 30)
(def neurons2 10)

(def syn0 (init-layer input neurons0))
;4x4
(def syn1 (init-layer neurons0 neurons1))
;4x30
(def syn2 (init-layer neurons1 neurons2))
;30x10
(def syn3 (init-layer neurons2 output))
;10x4
(defn l0 [x] (identity x))
;4x4
(defn l1 [x] (mm-map! (mm (l0 x) syn0) sigmoid))
;4x4 f[(4x4).(4x4)]
(defn l2 [x] (mm-map! (mm (l1 x) syn1) sigmoid))
;4x30 f[(4x4).(4x30)]
(defn l3 [x] (mm-map! (mm (l2 x) syn2) sigmoid))
;4x10 f[(4x30)(30x10)]
(defn l4 [x] (mm-map! (mm (l3 x) syn3) sigmoid))
;4x4 f[(4x10).(10x4)]
(defn l4_error [x y] (axpy -1 (l4 x) y))
;4x4
(defn l4_delta [x y] (h* (l4_error x y) (mm-map! (l4 x) training)))
;4x4 (4x4)*f'[(4x4)]
(defn l3_error [x y] (mm (l4_delta x y) (trans syn3)))
;4x10 (4x4).(4x10)
(defn l3_delta [x y] (h* (l3_error x y) (mm-map! (l3 x) training)))
;4x10 (4x10)*f'[(4x10)]
(defn l2_error [x y] (mm (l3_delta x y) (trans syn2)))
;4x30 (4x10).(10x30)
(defn l2_delta [x y] (h* (l2_error x y) (mm-map! (l2 x) training)))
;4x30 (4x30)*f'[(4x30)]
(defn l1_error [x y] (mm (l2_delta x y) (trans syn1)))
;4x4 (4x30).(30x4)
(defn l1_delta [x y] (h* (l1_error x y) (mm-map! (l1 x) training)))
;4x4 (4x4)*f'[(4x4)]
(println (l1_delta my-input my-output))

;(println (h* (trans (l3 my-input)) (l4_delta my-input my-output))(()))


