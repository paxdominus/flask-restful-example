import pytest

from app.contacts.models import ContactEmail, Contact


@pytest.yield_fixture()
def contact_email_1(db):
    return ContactEmail(email='johnwilliam@mail.com')


@pytest.yield_fixture()
def contact_email_2(db):
    return ContactEmail(email='williamjohn@mail.com')


@pytest.yield_fixture()
def contact_with_multiple_emails(db, contact_email_1, contact_email_2):
    contact = Contact(
        username='john',
        first_name='John',
        last_name='William',
    )

    contact_emails = [contact_email_1, contact_email_2]

    contact.emails.extend(contact_emails)
    with db.session.begin():
        db.session.add(contact)
        db.session.add_all(contact_emails)

    yield contact

    # Cleanup
    ContactEmail.query.filter(ContactEmail.contact == contact).delete()
    Contact.query.filter(Contact.id == contact.id).delete()


@pytest.yield_fixture()
def contact_with_a_email(db, temp_db_instance_helper):
    contact = Contact(
        username='sam',
        first_name='Sam',
        last_name='Henry',
    )

    contact_emails = [
        ContactEmail(email='samhenry@mail.com')
    ]

    contact.emails.extend(contact_emails)
    with db.session.begin():
        db.session.add(contact)
        db.session.add_all(contact_emails)

    yield contact

    # Cleanup
    ContactEmail.query.filter(ContactEmail.contact == contact).delete()
    Contact.query.filter(Contact.id == contact.id).delete()