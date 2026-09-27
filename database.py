import sqlite3
import hashlib


DB_NAME = "lms.db"


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    return sqlite3.connect(DB_NAME)


# =========================================================
# PASSWORD HASHING
# =========================================================

def hash_password(password):
    return hashlib.sha256(
        password.encode()
    ).hexdigest()


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    # -----------------------------------------------------
    # USERS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # -----------------------------------------------------
    # COURSES
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            teacher_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id)
                REFERENCES users(id)
        )
    """)

    # -----------------------------------------------------
    # MATERIALS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (course_id)
                REFERENCES courses(id)
        )
    """)

    # -----------------------------------------------------
    # QUIZZES
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            num_questions INTEGER NOT NULL,
            difficulty TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (course_id)
                REFERENCES courses(id)
        )
    """)

    # -----------------------------------------------------
    # QUIZ QUESTIONS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            option_a TEXT NOT NULL,
            option_b TEXT NOT NULL,
            option_c TEXT NOT NULL,
            option_d TEXT NOT NULL,
            correct_answer TEXT NOT NULL,
            explanation TEXT,
            FOREIGN KEY (quiz_id)
                REFERENCES quizzes(id)
        )
    """)

    # -----------------------------------------------------
    # QUIZ ATTEMPTS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            score INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (quiz_id)
                REFERENCES quizzes(id),
            FOREIGN KEY (student_id)
                REFERENCES users(id)
        )
    """)

    # -----------------------------------------------------
    # QUIZ ANSWERS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            selected_answer TEXT,
            is_correct INTEGER NOT NULL,
            FOREIGN KEY (attempt_id)
                REFERENCES quiz_attempts(id),
            FOREIGN KEY (question_id)
                REFERENCES quiz_questions(id)
        )
    """)


    # -----------------------------------------------------
    # ASSIGNMENTS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            title TEXT NOT NULL,
            questions TEXT NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            submission_date TEXT NOT NULL,
            FOREIGN KEY (teacher_id)
                REFERENCES users(id)
        )
    """)

    # -----------------------------------------------------
    # NOTIFICATIONS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            assignment_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id)
                REFERENCES users(id),
            FOREIGN KEY (assignment_id)
                REFERENCES assignments(id)
        )
    """)

    conn.commit()
    conn.close()


# =========================================================
# USER FUNCTIONS
# =========================================================

def create_user(
    name,
    email,
    password,
    role
):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            INSERT INTO users
            (name, email, password, role)
            VALUES (?, ?, ?, ?)
        """, (
            name,
            email,
            hash_password(password),
            role
        ))

        conn.commit()

        return True

    except sqlite3.IntegrityError:

        return False

    finally:

        conn.close()


def authenticate_user(
    email,
    password
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, email, role
        FROM users
        WHERE email = ?
        AND password = ?
    """, (
        email,
        hash_password(password)
    ))

    user = cursor.fetchone()

    conn.close()

    return user


# =========================================================
# COURSE FUNCTIONS
# =========================================================

def create_course(
    name,
    description,
    teacher_id
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO courses
        (name, description, teacher_id)
        VALUES (?, ?, ?)
    """, (
        name,
        description,
        teacher_id
    ))

    course_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return course_id


def get_courses():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            courses.id,
            courses.name,
            courses.description,
            users.name
        FROM courses
        JOIN users
            ON courses.teacher_id = users.id
        ORDER BY courses.id DESC
    """)

    courses = cursor.fetchall()

    conn.close()

    return courses


# =========================================================
# MATERIAL FUNCTIONS
# =========================================================

def add_material(
    course_id,
    filename,
    filepath
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO materials
        (course_id, filename, filepath)
        VALUES (?, ?, ?)
    """, (
        course_id,
        filename,
        filepath
    ))

    conn.commit()
    conn.close()


def get_materials(course_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            filename,
            filepath,
            uploaded_at
        FROM materials
        WHERE course_id = ?
        ORDER BY uploaded_at DESC
    """, (
        course_id,
    ))

    materials = cursor.fetchall()

    conn.close()

    return materials


# =========================================================
# QUIZ FUNCTIONS
# =========================================================

def create_quiz(
    course_id,
    title,
    num_questions,
    difficulty
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO quizzes
        (
            course_id,
            title,
            num_questions,
            difficulty
        )
        VALUES (?, ?, ?, ?)
    """, (
        course_id,
        title,
        num_questions,
        difficulty
    ))

    quiz_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return quiz_id


def add_quiz_question(
    quiz_id,
    question,
    option_a,
    option_b,
    option_c,
    option_d,
    correct_answer,
    explanation
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO quiz_questions
        (
            quiz_id,
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            correct_answer,
            explanation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        quiz_id,
        question,
        option_a,
        option_b,
        option_c,
        option_d,
        correct_answer,
        explanation
    ))

    conn.commit()
    conn.close()


def get_quizzes(course_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            course_id,
            title,
            num_questions,
            difficulty,
            created_at
        FROM quizzes
        WHERE course_id = ?
        ORDER BY created_at DESC
    """, (
        course_id,
    ))

    quizzes = cursor.fetchall()

    conn.close()

    return quizzes


def get_quiz(quiz_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            course_id,
            title,
            num_questions,
            difficulty,
            created_at
        FROM quizzes
        WHERE id = ?
    """, (
        quiz_id,
    ))

    quiz = cursor.fetchone()

    conn.close()

    return quiz


def get_quiz_questions(quiz_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            quiz_id,
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            correct_answer,
            explanation
        FROM quiz_questions
        WHERE quiz_id = ?
        ORDER BY id
    """, (
        quiz_id,
    ))

    questions = cursor.fetchall()

    conn.close()

    return questions


# =========================================================
# QUIZ ATTEMPT FUNCTIONS
# =========================================================

def create_quiz_attempt(
    quiz_id,
    student_id,
    score,
    total_questions
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO quiz_attempts
        (
            quiz_id,
            student_id,
            score,
            total_questions
        )
        VALUES (?, ?, ?, ?)
    """, (
        quiz_id,
        student_id,
        score,
        total_questions
    ))

    attempt_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return attempt_id


def add_quiz_answer(
    attempt_id,
    question_id,
    selected_answer,
    is_correct
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO quiz_answers
        (
            attempt_id,
            question_id,
            selected_answer,
            is_correct
        )
        VALUES (?, ?, ?, ?)
    """, (
        attempt_id,
        question_id,
        selected_answer,
        int(is_correct)
    ))

    conn.commit()
    conn.close()


def get_student_attempts(
    student_id
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            quiz_attempts.id,
            quiz_attempts.quiz_id,
            quiz_attempts.score,
            quiz_attempts.total_questions,
            quiz_attempts.attempted_at
        FROM quiz_attempts
        WHERE quiz_attempts.student_id = ?
        ORDER BY quiz_attempts.attempted_at DESC
    """, (
        student_id,
    ))

    attempts = cursor.fetchall()

    conn.close()

    return attempts


def get_quiz_attempts(
    quiz_id
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            quiz_attempts.id,
            quiz_attempts.student_id,
            quiz_attempts.score,
            quiz_attempts.total_questions,
            quiz_attempts.attempted_at
        FROM quiz_attempts
        WHERE quiz_attempts.quiz_id = ?
        ORDER BY quiz_attempts.attempted_at DESC
    """, (
        quiz_id,
    ))

    attempts = cursor.fetchall()

    conn.close()

    return attempts


# =========================================================
# NEW: STUDENT PERFORMANCE DETAILS
# =========================================================

def get_student_performance(
    student_id
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            quiz_attempts.id,
            quiz_attempts.quiz_id,
            quizzes.title,
            courses.name,
            quizzes.difficulty,
            quiz_attempts.score,
            quiz_attempts.total_questions,
            quiz_attempts.attempted_at
        FROM quiz_attempts

        JOIN quizzes
            ON quiz_attempts.quiz_id = quizzes.id

        JOIN courses
            ON quizzes.course_id = courses.id

        WHERE quiz_attempts.student_id = ?

        ORDER BY quiz_attempts.attempted_at DESC
    """, (
        student_id,
    ))

    performance = cursor.fetchall()

    conn.close()

    return performance


# =========================================================
# NEW: QUESTION-LEVEL PERFORMANCE
# =========================================================

def get_student_question_performance(
    student_id
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            quiz_attempts.id AS attempt_id,
            quizzes.id AS quiz_id,
            quizzes.title,
            courses.name,
            quiz_questions.id AS question_id,
            quiz_questions.question,
            quiz_questions.correct_answer,
            quiz_answers.selected_answer,
            quiz_answers.is_correct,
            quiz_attempts.attempted_at

        FROM quiz_answers

        JOIN quiz_attempts
            ON quiz_answers.attempt_id =
               quiz_attempts.id

        JOIN quiz_questions
            ON quiz_answers.question_id =
               quiz_questions.id

        JOIN quizzes
            ON quiz_attempts.quiz_id =
               quizzes.id

        JOIN courses
            ON quizzes.course_id =
               courses.id

        WHERE quiz_attempts.student_id = ?

        ORDER BY quiz_attempts.attempted_at DESC
    """, (
        student_id,
    ))

    performance = cursor.fetchall()

    conn.close()

    return performance


# =========================================================
# ASSIGNMENT FUNCTIONS
# =========================================================

def get_teacher_subject(teacher_id):
    """
    Returns the teacher's subject automatically from the
    course currently associated with the teacher.

    If the teacher has more than one course, the most
    recently created course is used for now.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM courses
        WHERE teacher_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (teacher_id,))

    row = cursor.fetchone()
    conn.close()

    return row[0] if row else None


def create_assignment(
    teacher_id,
    subject,
    title,
    questions,
    submission_date
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO assignments
        (
            teacher_id,
            subject,
            title,
            questions,
            submission_date
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        teacher_id,
        subject,
        title,
        questions,
        submission_date
    ))

    assignment_id = cursor.lastrowid

    # Notify all current student accounts.
    cursor.execute("""
        SELECT id
        FROM users
        WHERE role = 'Student'
    """)

    students = cursor.fetchall()

    teacher_name = ""
    cursor.execute("""
        SELECT name
        FROM users
        WHERE id = ?
    """, (teacher_id,))

    teacher_row = cursor.fetchone()

    if teacher_row:
        teacher_name = teacher_row[0]

    message = (
        f"New assignment posted: {title} | "
        f"Subject: {subject} | "
        f"Assigned by: {teacher_name} | "
        f"Submission Date: {submission_date}"
    )

    for student in students:
        cursor.execute("""
            INSERT INTO notifications
            (
                student_id,
                assignment_id,
                message
            )
            VALUES (?, ?, ?)
        """, (
            student[0],
            assignment_id,
            message
        ))

    conn.commit()
    conn.close()

    return assignment_id


def get_assignments_for_teacher(teacher_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            assignments.id,
            assignments.teacher_id,
            assignments.subject,
            assignments.title,
            assignments.questions,
            assignments.assigned_at,
            assignments.submission_date
        FROM assignments
        WHERE assignments.teacher_id = ?
        ORDER BY assignments.assigned_at DESC
    """, (teacher_id,))

    assignments = cursor.fetchall()
    conn.close()

    return assignments


def get_assignments_for_student(student_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            assignments.id,
            assignments.teacher_id,
            users.name,
            assignments.subject,
            assignments.title,
            assignments.questions,
            assignments.assigned_at,
            assignments.submission_date
        FROM assignments
        JOIN users
            ON assignments.teacher_id = users.id
        JOIN notifications
            ON notifications.assignment_id = assignments.id
        WHERE notifications.student_id = ?
        ORDER BY assignments.assigned_at DESC
    """, (student_id,))

    assignments = cursor.fetchall()
    conn.close()

    return assignments


def get_assignment(assignment_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            assignments.id,
            assignments.teacher_id,
            users.name,
            assignments.subject,
            assignments.title,
            assignments.questions,
            assignments.assigned_at,
            assignments.submission_date
        FROM assignments
        JOIN users
            ON assignments.teacher_id = users.id
        WHERE assignments.id = ?
    """, (assignment_id,))

    assignment = cursor.fetchone()
    conn.close()

    return assignment


# =========================================================
# NOTIFICATION FUNCTIONS
# =========================================================

def get_student_notifications(student_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            notifications.id,
            notifications.assignment_id,
            notifications.message,
            notifications.is_read,
            notifications.created_at,
            assignments.subject,
            assignments.title,
            assignments.submission_date,
            users.name
        FROM notifications
        JOIN assignments
            ON notifications.assignment_id = assignments.id
        JOIN users
            ON assignments.teacher_id = users.id
        WHERE notifications.student_id = ?
        ORDER BY notifications.created_at DESC
    """, (student_id,))

    notifications = cursor.fetchall()
    conn.close()

    return notifications


def get_unread_notification_count(student_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM notifications
        WHERE student_id = ?
        AND is_read = 0
    """, (student_id,))

    count = cursor.fetchone()[0]
    conn.close()

    return count


def mark_notification_read(notification_id, student_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE notifications
        SET is_read = 1
        WHERE id = ?
        AND student_id = ?
    """, (
        notification_id,
        student_id
    ))

    conn.commit()
    conn.close()


def mark_all_notifications_read(student_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE notifications
        SET is_read = 1
        WHERE student_id = ?
    """, (student_id,))

    conn.commit()
    conn.close()


# =========================================================
# DELETE ASSIGNMENT
# =========================================================

def delete_assignment(
    assignment_id,
    teacher_id
):
    """
    Delete an assignment only if it belongs to the
    currently logged-in teacher.

    Its student notifications are deleted first so that
    students do not keep seeing notifications for an
    assignment that no longer exists.
    """

    conn = get_connection()
    cursor = conn.cursor()

    # Make sure this assignment belongs to this teacher.
    cursor.execute("""
        SELECT id
        FROM assignments
        WHERE id = ?
        AND teacher_id = ?
    """, (
        assignment_id,
        teacher_id
    ))

    assignment = cursor.fetchone()

    if assignment is None:
        conn.close()
        return False

    # Remove notifications created for this assignment.
    cursor.execute("""
        DELETE FROM notifications
        WHERE assignment_id = ?
    """, (
        assignment_id,
    ))

    # Remove the assignment itself.
    cursor.execute("""
        DELETE FROM assignments
        WHERE id = ?
        AND teacher_id = ?
    """, (
        assignment_id,
        teacher_id
    ))

    conn.commit()
    conn.close()

    return True



# =========================================================
# DELETE COURSE
# =========================================================

def delete_course(course_id, teacher_id):
    """
    Permanently delete a course owned by the given teacher.
    Removes its materials, quizzes, quiz questions,
    quiz attempts, quiz answers, and the course record.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT id
            FROM courses
            WHERE id = ?
            AND teacher_id = ?
        """, (course_id, teacher_id))

        if cursor.fetchone() is None:
            return False

        cursor.execute("""
            SELECT id
            FROM quizzes
            WHERE course_id = ?
        """, (course_id,))

        quiz_ids = [row[0] for row in cursor.fetchall()]

        for quiz_id in quiz_ids:
            cursor.execute("""
                SELECT id
                FROM quiz_attempts
                WHERE quiz_id = ?
            """, (quiz_id,))

            attempt_ids = [row[0] for row in cursor.fetchall()]

            for attempt_id in attempt_ids:
                cursor.execute("""
                    DELETE FROM quiz_answers
                    WHERE attempt_id = ?
                """, (attempt_id,))

            cursor.execute("""
                DELETE FROM quiz_attempts
                WHERE quiz_id = ?
            """, (quiz_id,))

            cursor.execute("""
                DELETE FROM quiz_questions
                WHERE quiz_id = ?
            """, (quiz_id,))

        cursor.execute("""
            DELETE FROM quizzes
            WHERE course_id = ?
        """, (course_id,))

        cursor.execute("""
            DELETE FROM materials
            WHERE course_id = ?
        """, (course_id,))

        cursor.execute("""
            DELETE FROM courses
            WHERE id = ?
            AND teacher_id = ?
        """, (course_id, teacher_id))

        deleted = cursor.rowcount > 0

        conn.commit()
        return deleted

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

# =========================================================
# TEACHER: QUIZ PERFORMANCE REPORT
# =========================================================

# =========================================================
# TEACHER: QUIZ PERFORMANCE REPORT
# =========================================================

# =========================================================
# TEACHER: QUIZ PERFORMANCE REPORT
# =========================================================

def get_teacher_quiz_performance(
    teacher_id,
    course_id=None,
    quiz_id=None
):

    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT
            quiz_attempts.id AS attempt_id,
            quizzes.id AS quiz_id,
            quizzes.title AS quiz_title,
            courses.id AS course_id,
            courses.name AS course_name,
            users.id AS student_id,
            users.name AS student_name,
            users.email AS student_email,
            quiz_attempts.score,
            quiz_attempts.total_questions,
            quiz_attempts.attempted_at

        FROM quiz_attempts

        JOIN quizzes
            ON quiz_attempts.quiz_id = quizzes.id

        JOIN courses
            ON quizzes.course_id = courses.id

        JOIN users
            ON quiz_attempts.student_id = users.id

        WHERE courses.teacher_id = ?
    """

    parameters = [
        teacher_id
    ]

    if course_id is not None:

        query += """
            AND quizzes.course_id = ?
        """

        parameters.append(course_id)

    if quiz_id is not None:

        query += """
            AND quizzes.id = ?
        """

        parameters.append(quiz_id)

    query += """
        ORDER BY
            quiz_attempts.attempted_at DESC
    """

    cursor.execute(
        query,
        parameters
    )

    performance = cursor.fetchall()

    conn.close()

    return performance

# =========================================================
# AI TUTOR: PERSISTENT CHAT HISTORY
# =========================================================

import json


def create_ai_chat_session(
    user_id,
    course_id=None,
    title="New Chat"
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            course_id INTEGER,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)
                REFERENCES users(id),
            FOREIGN KEY (course_id)
                REFERENCES courses(id)
        )
    """)

    cursor.execute("""
        INSERT INTO ai_chat_sessions
        (
            user_id,
            course_id,
            title
        )
        VALUES (?, ?, ?)
    """, (
        user_id,
        course_id,
        title
    ))

    chat_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return chat_id


def save_ai_chat_message(
    chat_id,
    role,
    message,
    sources=None
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id)
                REFERENCES ai_chat_sessions(id)
        )
    """)

    sources_json = (
        json.dumps(sources)
        if sources
        else None
    )

    cursor.execute("""
        INSERT INTO ai_chat_messages
        (
            chat_id,
            role,
            message,
            sources
        )
        VALUES (?, ?, ?, ?)
    """, (
        chat_id,
        role,
        message,
        sources_json
    ))

    cursor.execute("""
        UPDATE ai_chat_sessions
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (
        chat_id,
    ))

    conn.commit()
    conn.close()


def get_ai_chat_sessions(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            course_id INTEGER,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)
                REFERENCES users(id),
            FOREIGN KEY (course_id)
                REFERENCES courses(id)
        )
    """)

    cursor.execute("""
        SELECT
            id,
            course_id,
            title,
            created_at,
            updated_at
        FROM ai_chat_sessions
        WHERE user_id = ?
        ORDER BY updated_at DESC
    """, (
        user_id,
    ))

    rows = cursor.fetchall()

    conn.close()

    return rows


def get_ai_chat_messages(chat_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id)
                REFERENCES ai_chat_sessions(id)
        )
    """)

    cursor.execute("""
        SELECT
            id,
            role,
            message,
            sources,
            created_at
        FROM ai_chat_messages
        WHERE chat_id = ?
        ORDER BY id ASC
    """, (
        chat_id,
    ))

    rows = cursor.fetchall()

    conn.close()

    messages = []

    for row in rows:

        sources = None

        if row[3]:

            try:
                sources = json.loads(
                    row[3]
                )

            except Exception:
                sources = None

        messages.append({
            "id": row[0],
            "role": row[1],
            "message": row[2],
            "sources": sources,
            "created_at": row[4]
        })

    return messages


def delete_ai_chat_session(
    chat_id,
    user_id
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM ai_chat_messages
        WHERE chat_id = ?
    """, (
        chat_id,
    ))

    cursor.execute("""
        DELETE FROM ai_chat_sessions
        WHERE id = ?
        AND user_id = ?
    """, (
        chat_id,
        user_id
    ))

    conn.commit()
    conn.close()