from create_event_tf_record import read_event_records

event_records_path = "/Users/ahmetkucuk/Documents/Research/solim_class/tfrecords/"

images, data, labels, label_txts = read_event_records(event_records_path, dataset_type="event_train")
print(len(images))