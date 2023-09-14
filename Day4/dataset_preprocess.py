# feature for long text dataset of QA

class QAdataset:

    def __init__(self,raw_datasets,tokenizer,args):
        self.dataset = raw_datasets
        self.tokenizer = tokenizer
        self.args = args
        self.pad_on_right = tokenizer.padding_side == "right"
        self.max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
        self.column_names = raw_datasets["train"].column_names
        self.cls_token_id = tokenizer.cls_token_id

    def prepare_train_features(self, examples):
        column_names = self.column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        # 用Truncation和padding對我們的資料進行標記，但是用stride來保持溢出的情況。這導致一個例子在上下文較長時可能給出幾個特徵，每個特徵的上下文都與前一個特徵的上下文有一定的重疊。
        tokenized_examples = self.tokenizer(
            examples[question_column_name if self.pad_on_right else context_column_name],
            examples[context_column_name if self.pad_on_right else question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.args.pad_to_max_length else False,
        )

        # 由於一個例子可能會給我們幾個特徵，如果它有一個長的上下文，我們需要一個從特徵到其相應的例子的映射。
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # offset_mapping將給我們一個原始上下文中的字符位置的映射。這將幫助我們計算起始位置和結束位置。
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # 獲得[CLS]的位置index
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.cls_token_id)

            # 抓取該例子相對應的序列（以獲得context，question）。
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 一個例子可以給出幾個跨度，這就是包含這個span的文本的例子的index。
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # 如果沒有給出答案，則將cls_index設為答案。
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 文本中答案的起始/結束token index。
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # 文本中span的起始token的index
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # 文本中span的結束token的index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # 檢測答案是否超出了span（特徵被標記為[CLS]）。
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # 否則将token_start_index和token_end_index移到答案的兩端。
                    # 注意：如果答案是最後一個字，我們可以在最後一個offset之後進行（邊緣情況）。
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self,examples):

        column_names = self.column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        # 用Truncation和padding對我們的資料進行標記，但是用stride來保持溢出的情況。這導致一個例子在上下文較長時可能給出幾個特徵，每個特徵的上下文都與前一個特徵的上下文有一定的重疊。
        tokenized_examples = self.tokenizer(
            examples[question_column_name if self.pad_on_right else context_column_name],
            examples[context_column_name if self.pad_on_right else question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.args.pad_to_max_length else False,
        )

        # 由於一個例子可能會給我們幾個特徵，如果它有一個長的上下文，我們需要一個從特徵到其相應的例子的映射。
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # 為了評估，我們需要將我們的預測轉換為上下文的子串，所以我們保留相應的example_id，我們將存儲偏移量的映射。
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # 抓取該例子相對應的序列（以獲得context，question）
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # 一個例子可以給出幾個跨度，這就是包含這個span的文本的例子的index。
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # 將不屬於上下文的offset_mapping設置為None，這樣就可以很容易地確定一個標記的位置是否屬於上下文的一部分。
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def generate_train_dataset(self):
        train_dataset = self.dataset["train"]

        if self.args.max_train_samples is not None:
            # 如果指定了，就挑選樣本
            train_dataset = train_dataset.select(range(self.args.max_train_samples))

        train_dataset = train_dataset.map(
            self.prepare_train_features,
            batched=True,
            num_proc=self.args.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        if self.args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(self.args.max_train_samples))

        return train_dataset

    def generate_eval_dataset(self):
        eval_examples = self.dataset["validation"]

        if self.args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(self.args.max_eval_samples))

        eval_dataset = eval_examples.map(
            self.prepare_validation_features,
            batched=True,
            num_proc=self.args.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        if self.args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(self.args.max_eval_samples))

        return eval_dataset
