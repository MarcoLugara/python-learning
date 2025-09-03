SELECT User.name, Member.role, Course.title FROM User
	JOIN Member ON User.id = Member.user_id
	JOIN Course ON Course.id = Member.course_id
	ORDER BY Course.title, Member.role DESC, User.name