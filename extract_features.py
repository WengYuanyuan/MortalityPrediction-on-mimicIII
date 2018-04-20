import csv
from pyspark import SparkContext, SparkConf
import pyspark_csv as pycsv
from pyspark.sql import SQLContext
from datetime import datetime
from get_text_feats import vectorize_text_feature

def is_infant(dob, icu_in):
    age = compute_age(dob, icu_in)
    return age >= 0 and age < 18

def is_dead(dod, icu_out):
    if not dod:
        return False
    return compute_diff(dod, icu_out) >= 0

def compute_age(dob, icu_in):
    seconds_diff = compute_diff(dob, icu_in)
    return int(seconds_diff / 31536000.0)

def compute_diff(date_strA, date_strB):
    dateA = __datetime(date_strA)
    dateB = __datetime(date_strB)
    return (dateB - dateA).total_seconds()

def __datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

def __datetimeLab(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')

def is_between_lab(eventtime, icu_in, icu_out):
    if not eventtime:
        return True
    eventtime = __datetimeLab(eventtime)
    icu_in = __datetimeLab(icu_in.split(' ')[0])
    icu_out = __datetimeLab(icu_out.split(' ')[0])
    return (eventtime - icu_in).total_seconds() >= 0 and (icu_out - eventtime).total_seconds() >= 0

def is_between(eventtime, icu_in, icu_out):
    if not eventtime:
        return True
    return (compute_diff(icu_in, eventtime) >= 0
            and compute_diff(eventtime, icu_out) >= 0)

def unpack(list_result_iterables):
    r = []
    for result_iterable in list_result_iterables:
        for res in result_iterable:
            r.append(res)
    return r

def encode_set(event_set, length):
    encoded_cpt = [0] * length
    for e in event_set:
        encoded_cpt[e[1]] = 1
    return encoded_cpt

sc = SparkContext()
sc.addPyFile('pyspark_csv.py')

all_patients = sc.textFile("data/PATIENTS.csv").cache()
icu_visits = sc.textFile("data/ICUSTAYS.csv").cache()
cpt_events = sc.textFile("data/CPTEVENTS.csv").cache()
note_events = sc.textFile("data/NOTEEVENTS.csv").cache()
lda_events = sc.textFile("data/LDAEVENTS.csv").cache()
lab_events = sc.textFile("data/LABEVENTS.csv").cache()

sc = SQLContext(sc)

all_patients = pycsv.csvToDataFrame(sc, all_patients, parseDate=False).rdd
icu_visits = pycsv.csvToDataFrame(sc, icu_visits, parseDate=False).rdd
cpt_events = pycsv.csvToDataFrame(sc, cpt_events, parseDate=False).rdd
note_events = pycsv.csvToDataFrame(sc, note_events, parseDate=False).rdd
lda_events = pycsv.csvToDataFrame(sc, lda_events, parseDate=False).rdd
lab_events = pycsv.csvToDataFrame(sc, lab_events, parseDate=False).rdd

icu_patients = all_patients.keyBy(lambda p: p.SUBJECT_ID).join(icu_visits.keyBy(lambda v: v.SUBJECT_ID))
children_data = icu_patients.filter(lambda ip: is_infant(ip[1][0].DOB, ip[1][1].INTIME))
children_ids = children_data.map(lambda c: c[0]).collect()
visit_ids = children_data.map(lambda c: c[1][1].ICUSTAY_ID)

# AGE AND GENDER FEATURES
visits_age_gender = children_data.map(lambda c: (c[1][1].ICUSTAY_ID, compute_age(c[1][0].DOB, c[1][1].INTIME), c[1][0].GENDER)).keyBy(lambda c: c[0]).map(lambda c: (c[0], (c[1][1], c[1][2])))

# DEATH LABEL
visits_is_dead = children_data.map(lambda c: (c[1][1].ICUSTAY_ID, is_dead(c[1][0].DOD, c[1][1].OUTTIME))).keyBy(lambda c: c[0]).map(lambda c: (c[0], (c[1][1])))

# CPT EVENTS FEATURE
# the code below currently only computes counts of distinct cpt events per icu stay. It turns out these events are rather sparse for our <18 patients, so we're not doing the 1-hot encoding now as the majority of them would be empty vectors of length ~78
visits_cpt = children_data.leftOuterJoin(cpt_events.keyBy(lambda cpt: cpt.SUBJECT_ID)).filter(lambda e: not e[1][1] or is_between(e[1][1].CHARTDATE, e[1][0][1].INTIME, e[1][0][1].OUTTIME))
visits_cpt = visits_cpt.map(lambda cpt: (cpt[1][0][1].ICUSTAY_ID, cpt[1][1].CPT_CD if cpt[1][1] else None)).groupByKey().mapValues(lambda v: len([cpt for cpt in v if cpt]))
visits_cpt = visits_cpt.keyBy(lambda cpt: cpt[0]).map(lambda c: (c[0], (c[1][1])))

# LDA FEATURE - this will be run on the server post-draft
# this computes lda for notes
#visits_notes = children_data.leftOuterJoin(note_events.keyBy(lambda note: note.SUBJECT_ID)).filter(lambda n: not n[1][1] or is_between_notes(n[1][1].CHARTDATE, n[1][0][1].INTIME, n[1][0][1].OUTTIME))
#visits_notes = visits_notes.map(lambda notes: (notes[1][0][1].ICUSTAY_ID, vectorize_text_feature(notes[1][1].TEXT) if notes[1][1] else None))
#print(visits_notes.count())
#print(visits_notes.first())

# for now, we are just using a pandas-computed lda featureset
visits_lda = lda_events.keyBy(lambda l: l.ICUSTAY_ID).map(lambda l: (l[1].ICUSTAY_ID, list(l[1][2:])))
num_topics = len(visits_lda.first()[1])
visits_nonotes = visit_ids.subtract(visits_lda.map(lambda l: l[0]))
visits_nonotes = visits_nonotes.map(lambda l: (l, [0] * num_topics))
visits_lda = visits_lda.union(visits_nonotes)

# LAB EVENTS FEATURE
# below is the code for obtaining a 1-hot encoding of abnormal labevents per icu stay. 6044/8200 icu stays have at least one 'abnormal' lab event
visits_lab = children_data.join(lab_events.keyBy(lambda lab: lab.SUBJECT_ID)).filter(lambda e: is_between_lab(e[1][1].CHARTTIME, e[1][0][1].INTIME, e[1][0][1].OUTTIME))
lab_code_map = dict(visits_lab.map(lambda lab: lab[1][1].ITEMID).distinct().zipWithIndex().collect())
visits_lab = visits_lab.map(lambda lab: (lab[1][0][1].ICUSTAY_ID, lab_code_map[lab[1][1].ITEMID], lab[1][1].FLAG)).filter(lambda lab: lab[2] == 'abnormal').groupBy(lambda l: l[0]).mapValues(lambda v: encode_set(set(v), len(lab_code_map.keys())))
visits_nolabs = visit_ids.subtract(visits_lab.map(lambda l: l[0]))
visits_nolabs = visits_nolabs.map(lambda l: (l, ([0] * len(lab_code_map.keys()))))
visits_lab = visits_lab.keyBy(lambda v: v[0]).map(lambda v: (v[0], (v[1][1])))
visits_lab = visits_lab.union(visits_nolabs)

# AGGREGATE FEATURES
# NOTE: features_RDD is an RDD of 8200 tuples of the form (ICUSTAY_ID, List[feats])
# One instance looks like this:
# (ICUSTAY_ID, [(age, gender), is_dead, cpt_event_Count, [1-hot-encoded lab_events], [LDA topic values]])
features_RDD = visits_age_gender.groupWith(visits_is_dead, visits_cpt, visits_lab,visits_lda).mapValues(list).map(lambda r: (r[0], unpack(r[1])))
print(features_RDD.first())


