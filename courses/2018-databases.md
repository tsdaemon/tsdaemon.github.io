---
layout: page
title: Introduction to databases
permalink: /courses/2018-databases/
---

*Wow, this one was pretty tough! I've just finished my MSc and started to work
at my first job as a data scientist. And I thought that teaching
other is much simpler than being taught. That was the biggest mistake I've
ever made. Amount of work to prepare lectures and homework was really a challenge for me,
so I spent almost all weekends during spring 2018, preparing lecture notes and reviewing
homework.*

Here a couple of lessons I have learned from this course:
1. I have selected C# and .NET as the environment for course assignments, but my students
were barely able to comprehend it. I can agree, that they should learn it even if
it more complex than Python; but I also **already bored of it after 5 years of .NET development**.
I guess I will use Python or Go for the next year assignments.
1. Lectures are really much more interactive when you mix them with ad hoc practical
assignments. On the lecture about aggregation, I have **prepared a Kaggle dataset and we
have a lot of fun** querying it for different data slices.
2. I prepared a functional test to check students solutions automatically, yet I **still needed
to clone their code and start the test manually**. This was a big overhead on these few actions.
I will use Travis or something like that to automate this process next year.
    * Also, I made the test available for students and they really overused it.
    Many of them **didn't even bother to test** their solutions manually.

# Lecture notes

1. [Course overview](https://drive.google.com/open?id=1uw80u6q5_aGZaS6W7rwPXSyjunraVtfP)

2. [DBMS 1](https://drive.google.com/open?id=1pDrthqSBUuv7bzvJ-hO1ASNHaJINDQIM)
    * DBMS vs filesystem
    * Data abstraction
    * Physical independence
    * Relational data model

3. [Query 1](https://drive.google.com/file/d/1G2Vq-qA-cFRbZ4ffEVq-mJBpsU9rMG9G/view?usp=sharing)
    * Data manipulation languages
    * SQL: SELECT, FROM, AS, WHERE, ORDER BY, LIMIT
    * Relation algebra: selection, projection

4. [Modeling 1](https://drive.google.com/file/d/1CuEzfFLpAR8XAI9yZqzyTm1W5qv_ds18/view?usp=sharing)
    * Data definition language
    * Conceptual vs logical vs physical data models
    * SQL data types
    * Constraints
    * Keys
    * SQL: CREATE

5. [Writing 1](https://drive.google.com/open?id=16L3kcOI2KXt6Qv8XxK4im8atvKjgbJmE)
    * SQL: INSERT, UPDATE, DELETE.

6. [Application 1](https://drive.google.com/open?id=16TGHgDuYR9vxkkK8kneR0_-bogRTiLx4)
    * System.Data: IDbConnection, IDbCommand, IDbReader, IDbDataAdapter
    * ADO.NET

7. [Not Only SQL 1](https://drive.google.com/open?id=1DiviZ1h6TJQ0XfA9QwL9ImB1t2FTc_5n)
    * Why NoSQL: impendance mismatch, speed, big data
    * Why SQL: single language, static typing, integrity control

8. [DBMS 2](https://drive.google.com/open?id=1icp07eYd4XZ2AbiXKcNf5pqdpQhwYOZ6)
    * Backups

9. [Modeling 2](https://drive.google.com/open?id=1kECUAXRudGkMr-Ng_22zo7lWO9FuLCIN)
    * Associations
      * One-to-many
      * Many-to-many
      * One-to-one
    * Foreign key

10. [Query 2](https://drive.google.com/open?id=1hNtFUnNafS6AopFDFRb1uejZ_bwpTjgh)
    * Cartesian product
    * Join: inner, left, right, outer
    * Views

11. [Application 2](https://drive.google.com/open?id=1NPBieAXMT1Pe5MSQSdXzFdS6L72tfjZf)
    * ORM
    * Dapper.NET
    * Linq2SQL
    * EntityFramework

12. [DBMS 3](https://drive.google.com/open?id=1NIDMC-fZgEQFr0tUmDwefnzTOlLaU1Cy)
    * Migrations

13. [Query 3](https://drive.google.com/open?id=1L4VCsdKmyqTkjZGB_loFbD066_WHynHN)
    * Aggregation functions
    * GROUP BY
    * HAVING

14. [Modeling 3](https://drive.google.com/open?id=17p1QXXZUEJ6KAY_jrqeHj5ZJoAkBSS0-)
    * Normalization
    * First normal form
    * Second normal form
    * Third normal form
    * Normalized vs de-normalized

15. [Writing 3](https://drive.google.com/open?id=1Y6kJicCvBlFKJHKhm6yL2-qYMwhe9-0g)
    * Transactions
    * ACID

16. [Application 3](https://drive.google.com/open?id=1rChrMWgdAV5TjmnW54OdInnk-ILUdiaa)
    * SOLID
    * Layered architecture
    * Dependency injection
    * IRepository

17. [DBMS 4](https://drive.google.com/open?id=1CK2DcFe0vIVtzpbGoGtI4h136bSyHXj8)
    * Storage level
    * DBMS file structure
    * Database buffer
    * Journaling
    * Database engines

18. [Modeling 4](https://drive.google.com/open?id=19cNoRaqUt0qGseEHyDi6biBqRM7JlwL5)
    * Query optimization
    * Data structures overview
    * indexes
    * Fulltext search
    * JOIN strategies

19. [Not Only SQL 4](https://drive.google.com/open?id=1LLRZFdJVH06vquPYXOIsLwfUeQcYqn9-)
    * NoSQL categories
    * Key-value storage
    * Wide column storage
    * Document storage
    * Graph storage
