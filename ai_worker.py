"""Background worker for AI-LMS generation."""
from __future__ import annotations
import json, logging, sys, traceback
from pathlib import Path
from ai_tutor import get_ai_tutor
from learning_mode import contextualize_question, build_learning_history, is_follow_up
from youtube_recommender import recommend_youtube
BASE_DIR=Path(__file__).resolve().parent
LOG_DIR=BASE_DIR/'logs'; LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(filename=str(LOG_DIR/'ai_worker.log'),level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')

def normalize(raw):
    if isinstance(raw,str): return {'answer':raw,'sources':[],'diagram':None}
    if not isinstance(raw,dict): return {'answer':str(raw),'sources':[],'diagram':None}
    if 'answer' in raw: return {'answer':str(raw.get('answer','')),'sources':raw.get('sources',[]) or [],'diagram':raw.get('diagram')}
    nested=raw.get('result')
    if isinstance(nested,dict): return normalize(nested)
    return {'answer':str(raw.get('content') or raw.get('response') or raw.get('text') or 'I could not generate an answer.'),'sources':raw.get('sources',[]) or [],'diagram':raw.get('diagram')}

def main(request_path,result_path):
    data={}
    result={'ok':False,'answer':'I could not generate the answer. Please try again.','sources':[],'youtube':None,'learning_mode':False}
    try:
        data=json.loads(Path(request_path).read_text(encoding='utf-8'))
        question=str(data.get('question','')).strip()
        history=str(data.get('history_text',''))
        attachment_text=str(data.get('attachment_text',''))
        learning_mode=bool(data.get('learning_mode',False))
        if not question: raise ValueError('Question is empty')

        follow_up=is_follow_up(question,history)

        # Keep the exact question in answer history. Retrieval context is separate.
        if learning_mode:
            answer_history=build_learning_history(history,attachment_text,question)
        else:
            answer_history=history
            if attachment_text.strip():
                answer_history += '\n\nUPLOADED MATERIAL FOR THIS QUESTION:\n'+attachment_text[-12000:]

        # AITutor owns contextual retrieval. Pass the exact student question
        # so answer intent and answer size are detected from the real wording.
        raw=get_ai_tutor().ask(
            question=question,
            course_id=None,
            material_id=None,
            history=answer_history,
        )
        normalized=normalize(raw)

        youtube=None
        try:
            youtube=recommend_youtube(question,normalized.get('answer',''),history)
        except Exception:
            logging.exception('YouTube recommendation failed; answer remains valid')

        result={'ok':True,**normalized,'youtube':youtube,'learning_mode':learning_mode,'follow_up':follow_up}
        logging.info('Generation complete | learning=%s follow_up=%s q=%r',learning_mode,follow_up,question)
    except Exception as exc:
        logging.error('AI generation failed: %s\n%s',exc,traceback.format_exc())
        result={'ok':False,'answer':'I could not generate the answer. Please try again.','sources':[],'youtube':None,'learning_mode':bool(data.get('learning_mode',False)),'detail':str(exc)}
    Path(result_path).write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    if len(sys.argv)!=3: raise SystemExit('Usage: python ai_worker.py REQUEST_JSON RESULT_JSON')
    main(sys.argv[1],sys.argv[2])
