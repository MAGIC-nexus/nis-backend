from sqlalchemy import Column, Integer, String, DateTime, Float, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, composite, scoped_session, sessionmaker

# import pandasql

__author__ = 'rnebot'

DBSession = scoped_session(sessionmaker())  # TODO - Is this thread safe ??


class BaseMixin(object):
    # query = DBSession.query_property()
    # id = Column(Integer, primary_key=True)
    # @declared_attr
    # def __tablename__(cls):
    # return cls.__name__.lower()
    pass


ORMBase = declarative_base(cls=BaseMixin)


