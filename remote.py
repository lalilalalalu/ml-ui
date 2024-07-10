
smt = """
           SELECT sqml_train(
               '{name}',
               '{type}',
               '{algo}',
               '{table}',
               '{target}'
           )
           AS
           training;
           """.format(name=self.ml_config['name'], type=self.ml_config['type'].name,
                      algo=self.ml_config['algo'],
                      table=self.ml_config['table'], target=self.ml_config['target'][0])
print(smt)
results = self.conn.execute(smt).fetchone()[0]
try:
    self.query_one("#sc_pre", ScrollableContainer)
except NoMatches:
    self.mount(ScrollableContainer(id="sc_pre"))
sc_pre = self.query_one("#sc_pre", ScrollableContainer)
try:
    self.query_one("#pretty_result", Pretty)
except NoMatches:
    sc_pre.mount(Pretty("", id="pretty_result"))
pretty_result = self.query_one("#pretty_result", Pretty)
pretty_result.update(results)
pretty_result.border_title = "Training Output:"
