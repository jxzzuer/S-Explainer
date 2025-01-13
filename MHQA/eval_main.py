import json
import logging
import os
from eval_odqa import ODQAEval
from eval_utils import evaluate, evaluate_w_sp_facts, convert_qa_sp_results_into_hp_eval_format


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        odqa = ODQAEval()

        if odqa.args.sequential_sentence_selector_path is not None:
            tfidf_retrieval_output, selector_output, reader_output, sp_selector_output = odqa.eval()
            if odqa.args.sp_eval:
                predictions = convert_qa_sp_results_into_hp_eval_format(
                    reader_output, sp_selector_output, odqa.args.db_path)
                results = evaluate_w_sp_facts(
                    odqa.args.eval_file_path_sp, predictions, odqa.args.sampled)
            else:
                results = evaluate(odqa.args.eval_file_path, reader_output)
            logger.info("Evaluation Results: %s", results)
        else:
            tfidf_retrieval_output, selector_output, reader_output = odqa.eval()
            results = evaluate(odqa.args.eval_file_path, reader_output)
            logger.info("EM: %s, F1: %s", results['exact_match'], results['f1'])


        if odqa.args.tfidf_results_save_path:
            save_results(os.path.join(odqa.args.tfidf_results_save_path, "tfidf_results.json"), tfidf_retrieval_output, 'TFIDF Retrieval')

        if odqa.args.selector_results_save_path:
            save_results(os.path.join(odqa.args.selector_results_save_path, "selector_results.json"), selector_output, 'graph-based Retrieval')

        if odqa.args.reader_results_save_path:
            save_results(os.path.join(odqa.args.reader_results_save_path, "reader_results.json"), reader_output, 'reader')

        if odqa.args.sequence_sentence_selector_save_path:
            save_results(os.path.join(odqa.args.sequence_sentence_selector_save_path, "sequence_sentence_selector_results.json"), sp_selector_output, 'sentence selector')

    except Exception as e:
        logger.error("An error occurred: %s", e, exc_info=True)

def save_results(path, data, description):
    try:
        logger.info('#### save %s results to %s ####', description, path)
        with open(path, "w") as writer:
            writer.write(json.dumps(data, indent=4) + "\n")
        logger.info('%s results saved successfully', description)
    except IOError as e:
        logger.error("Failed to save %s results: %s", description, e)

if __name__ == "__main__":
    main()
