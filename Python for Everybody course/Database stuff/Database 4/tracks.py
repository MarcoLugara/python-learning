import sqlite3

conn = sqlite3.connect('trackdb.sqlite')
cur = conn.cursor()

# Make some fresh tables using executescript()
cur.executescript('''
DROP TABLE IF EXISTS Artist;
DROP TABLE IF EXISTS Album;
DROP TABLE IF EXISTS Track;
DROP TABLE IF EXISTS Genre;

CREATE TABLE Artist (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name    TEXT UNIQUE
);

CREATE TABLE Genre (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name    TEXT UNIQUE
);

CREATE TABLE Album (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    artist_id  INTEGER,
    title   TEXT UNIQUE
);

CREATE TABLE Track (
    id  INTEGER NOT NULL PRIMARY KEY 
        AUTOINCREMENT UNIQUE,
    title TEXT  UNIQUE,
    album_id  INTEGER,
    genre_id  INTEGER,
    len INTEGER, rating INTEGER, count INTEGER
);
''')

handle = open('tracks.csv')

# Another One Bites The Dust, Queen, Greatest Hits, 55, 100, 217103
#   0                                      ,  1       , 2                   , 3  ,  4  , 5

for line in handle:
    line = line.strip();
    #print(line)
    #print(type(line))
    pieces = line.split(',')
    #print(pieces)

    #We check if all the lists are made correctly, i.e. right number of entries for each, should be 6
    if len(pieces) < 7 :
        continue

    # Another One Bites The Dust, Queen, Greatest Hits, 55, 100, 217103, Rock
    #   0                                      ,  1       , 2                   , 3  ,  4  , 5,        , 6

    title = pieces[0]
    artist = pieces[1]
    album = pieces[2]
    length = pieces[3]
    rating = pieces[4]
    count = pieces[5]
    genre = pieces[6]

    print(title, artist, album, length, rating, count, genre)

    #We then add our values to the table
    #We start with Artist
    cur.execute('''INSERT OR IGNORE INTO Artist (name) 
        VALUES ( ? )''', ( artist, ) )
    cur.execute('SELECT id FROM Artist WHERE name = ? ', (artist, ))
    artist_id = cur.fetchone()[0]

    #We continue with Genre
    cur.execute('''INSERT OR IGNORE INTO Genre (name) 
            VALUES ( ? )''', (genre,))
    cur.execute('SELECT id FROM Genre WHERE name = ? ', (genre,))
    genre_id = cur.fetchone()[0]

    #Now, we do Album, using artist_id
    cur.execute('''INSERT OR IGNORE INTO Album (title, artist_id) 
        VALUES ( ?, ? )''', ( album, artist_id ) )
    cur.execute('SELECT id FROM Album WHERE title = ? ', (album, ))
    album_id = cur.fetchone()[0]

    cur.execute('''INSERT OR REPLACE INTO Track
        (title, album_id, genre_id, len, rating, count) 
        VALUES ( ?, ?, ?, ?, ?, ? )''',
        ( title , album_id, genre_id, length, rating, count ) )

    conn.commit()
#We have created our pretty tables.

#We, now, wish to create a new table by joining data
    cur.execute('''
    DROP VIEW IF EXISTS Finaltable '''
    )
    cur.execute('''
    CREATE VIEW FinalTable AS
    SELECT Track.title, Artist.name, Album.title, Genre.name
        FROM Track
            JOIN Genre ON Track.genre_id = Genre.id
            JOIN Album ON Track.album_id = Album.id
            JOIN Artist ON Album.artist_id = Artist.id
        ---here we make the joined tables to the filter the result
    ORDER BY Artist.name LIMIT 3
    ---here we filter the first 3 results in the new table, where they are ordered by artist name
    '''
    )
    conn.commit()

